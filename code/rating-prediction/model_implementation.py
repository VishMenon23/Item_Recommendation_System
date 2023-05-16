import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import Reader, Dataset
from surprise import SVD, accuracy, SVDpp, SlopeOne, BaselineOnly, CoClustering, KNNBasic
import data_processing
import time
import pickle
from sklearn.metrics import ndcg_score


def convert_train_test_dataframe(training_dataframe, testing_dataframe):
	training_dataframe = training_dataframe.iloc[:, :-1]
	testing_dataframe = testing_dataframe.iloc[:, :-1]
	reader = Reader(rating_scale=(0,5))
	trainset = Dataset.load_from_df(training_dataframe[['reviewerID', 'asin', 'overall']], reader)
	testset = Dataset.load_from_df(testing_dataframe[['reviewerID', 'asin', 'overall']], reader)
	trainset = trainset.construct_trainset(trainset.raw_ratings)
	testset=testset.construct_testset(testset.raw_ratings)
	return([trainset,testset])

def evaluate_error(actual_ratings, estimate_ratings):
	ratings = np.array(actual_ratings)
	estimate = np.array(estimate_ratings)
	rmse = np.sqrt(np.sum(np.square(np.subtract(ratings, estimate)))/np.size(ratings))
	mae = np.sum(np.abs(np.subtract(ratings, estimate)))/np.size(ratings)
	return rmse, mae

def svdalgorithm(trainset, testset):
  print("\n" + "-" *5 + " SVD algorithm " + "-" *5)
  algo = SVD()
  algo.fit(trainset)
  predictions = algo.test(testset)
  print(predictions[:10])
  rmse = accuracy.rmse(predictions)
  mae = accuracy.mae(predictions)
  return rmse, mae, predictions

def svdpp(trainset, testset):
	# Matrix factorization - SVD++
  print("\n" + "-" *5 + " SVD++ algorithm" + "-" *5)
  algo = SVDpp()
  algo.fit(trainset)
  predictions = algo.test(testset)
  print(predictions[:10])
  rmse = accuracy.rmse(predictions)
  mae = accuracy.mae(predictions)
  return rmse, mae, predictions

def coClustering(trainset, testset):
	# CoClustering
  print("\n" + "-" *5 + " CoClustering algorithm" + "-" *5)
  algo = CoClustering()
  algo.fit(trainset)
  predictions = algo.test(testset)
  print(predictions[:10])
  rmse = accuracy.rmse(predictions)
  mae = accuracy.mae(predictions)
  return rmse, mae, predictions


def precision_recall_evaluation(predictions, threshold=3.5):
    user_predict_true = defaultdict(list)
    for reviewer_id, product_id, true_rating, predicted_rating, _ in predictions:
        user_predict_true[reviewer_id].append((predicted_rating, true_rating))

    precisions = dict()
    recalls = dict()
    for reviewer_id, user_ratings in user_predict_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)       # Sort user ratings by estimated value
        no_of_relevant_items = sum((true_rating >= threshold) for (predicted_rating, true_rating) in user_ratings) # Number of relevant items
        no_of_recommended_items = sum((predicted_rating >= threshold) for (predicted_rating, true_rating) in user_ratings[:10])  # Number of recommended items in top 10
        no_of_relevant_and_recommended_items = sum(((true_rating >= threshold) and (predicted_rating >= threshold)) for (predicted_rating, true_rating) in user_ratings[:10])   # Number of relevant and recommended items in top 10
        precisions[reviewer_id] = no_of_relevant_and_recommended_items / no_of_recommended_items if no_of_recommended_items != 0 else 1    # Precision: Proportion of recommended items that are relevant
        recalls[reviewer_id] = no_of_relevant_and_recommended_items / no_of_relevant_items if no_of_relevant_items != 0 else 1        # Recall: Proportion of relevant items that are recommended

    # Averaging the values for all users
    average_precision=sum(precision for precision in precisions.values()) / len(precisions)
    average_recall=sum(recall for recall in recalls.values()) / len(recalls)
    F_score=(2*average_precision*average_recall) / (average_precision + average_recall)
    
    return [average_precision, average_recall, F_score]

#Writes the predictions of all approaches to text files
def model_impl(trainset, testset, exectution_mode = "Test", weights = []):
	if exectution_mode == "Train":
		start_time = time.time()
		rmse_list = []
		mae_list = []

		rmse, mae, svd_pred = svdalgorithm(trainset, testset)
		print(f'SVD: rmse - {rmse}, mae - {mae}')
		file = open('Predictions/svd_pred.txt', 'wb')
		pickle.dump(svd_pred, file)
		file.close()
		rmse_list.append(rmse)
		mae_list.append(mae)
		print("Elapsed Time: ", time.time() - start_time)

		rmse, mae, svdplus_pred = svdpp(trainset, testset)
		print(f'SVD++: rmse - {rmse}, mae - {mae}')
		file = open('Predictions/svdplus_pred.txt', 'wb')
		pickle.dump(svdplus_pred, file)
		file.close()
		rmse_list.append(rmse)
		mae_list.append(mae)
		print("Elapsed Time: ", time.time() - start_time)
		
		rmse, mae, clustering_pred = coClustering(trainset, testset)
		print(f'Co-clustering: rmse - {rmse}, mae - {mae}')
		file = open('Predictions/clustering_pred.txt', 'wb')
		pickle.dump(clustering_pred, file)
		file.close()
		rmse_list.append(rmse)
		mae_list.append(mae)
		print("Elapsed Time: ", time.time() - start_time)

		file = open('Predictions/mae_list.txt', 'wb')
		pickle.dump(mae_list, file)
		file.close()

		file = open('Predictions/rmse_list.txt', 'wb')
		pickle.dump(rmse_list, file)
		file.close()

	if exectution_mode == "Test":
		predictions_all = []

		file = open('Predictions/svd_pred.txt', 'rb+')
		svd_predictions = pickle.load(file)
		[precision, recall, F_score] = precision_recall_evaluation(svd_predictions, threshold=3.5)

		print("\n" + "-" *5 + " SVD - Item Prediction " + "-" *5)
		print("Precision: ", precision)
		print("Recall: ", recall)
		print("F-Score: ",F_score)
		predictions_all.append(svd_predictions)
		file.close()

		file = open('Predictions/svdplus_pred.txt', 'rb+')
		svdpp_predictions = pickle.load(file)
		[precision, recall, F_score] = precision_recall_evaluation(svdpp_predictions, threshold=3.5)

		print("\n" + "-" *5 + " SVD++ - Item Prediction " + "-" *5)
		print("Precision: ", precision)
		print("Recall: ", recall)
		print("F-Score: ",F_score)
		predictions_all.append(svdpp_predictions)
		file.close()

		file = open('Predictions/clustering_pred.txt', 'rb+')
		coclustering_predictions = pickle.load(file)
		[precision, recall, F_score] = precision_recall_evaluation(coclustering_predictions, threshold=3.5)

		print("\n" + "-" *5 + " Co-clustering - Item Prediction " + "-" *5)
		print("Precision: ", precision)
		print("Recall: ", recall)
		print("F-Score: ",F_score)
		predictions_all.append(svd_predictions)
		file.close()

		file = open('Predictions/mae_list.txt', 'rb+')
		mae_list = pickle.load(file)
		file.close()

		file = open('Predictions/rmse_list.txt', 'rb+')
		rmse_list = pickle.load(file)
		file.close()

		actual_ratings = []
		estimate_arr = []

		for p in predictions_all[1]:
			actual_ratings.append(p[2])

		for i, predictions in enumerate(predictions_all):
			estimate_arr.append([])
			for p in predictions:
				estimate_arr[i].append(p[3])

		if len(weights) == 0:
			total = 0
			for i, (e,f) in enumerate(zip(rmse_list, mae_list)):
				if i in [0, 1, 2, 3, 4, 5]:
					total += (1)/((e) ** 1)

			for i, (e,f) in enumerate(zip(rmse_list, mae_list)):
				if i in [0, 1, 2, 3, 4, 5]:
					weights.append((1)/(((e) ** 1) * total))
				else:
					weights.append(0)

			hybrid_estimates = np.zeros(np.asarray(estimate_arr[0]).shape)

			for i, estimate in enumerate(estimate_arr):
				hybrid_estimates += np.multiply(estimate, weights[i])

		print(weights)

		hybrid_predictions = []

		for p, h in zip(predictions_all[0], hybrid_estimates):
			hybrid_predictions.append((p[0], p[1], p[2], h, p[4]))

		rmse, mae = evaluate_error(actual_ratings, hybrid_estimates)
		[precision, recall, F_score] = precision_recall_evaluation(hybrid_predictions, threshold=3.5)

		print("\n" + "-" *5 + " Hybrid algorithm " + "-" *5)
		print("RMSE: ", rmse)
		print("MAE: ", mae)
		print("Precision: ", precision)
		print("Recall: ", recall)
		print("F-Score: ",F_score)

		print(str(rmse) + "\t" + str(mae) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(F_score))
	
if __name__ == "__main__":
	trainset = []
	testset = []
	execution_mode = "Test"
	weights = []
	if execution_mode == "Train":
		df_train, df_test = data_processing.get_train_test_data(new_sample = True)
		trainset, testset = convert_train_test_dataframe(df_train, df_test)
	model_impl(trainset, testset, execution_mode, weights)