import os
import subprocess
from statistics import mean
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from modAL import ActiveLearner
import pandas as pd
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling
from sklearn import svm, preprocessing
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from sklearn.metrics import balanced_accuracy_score, accuracy_score, average_precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
import shutil
import cv2
from glob import glob
import json
import sqlite3
from sqlite3 import Error


def read_config(config_file):
    """
    read_config - reads the configuration file for the script
    :param config_file: path to the config file
    """
    with open(config_file) as f:
        for line in f:
            (key, val) = line.split('=')
            arguments_dictionary[key] = val.strip('\n')


def run_cellprofiler():
    """
    run_cellprofiler - runs the command for executing the  CellProfiler pipeline on the sample of images with metadata
    """
    cellprofiler = arguments_dictionary[app_dir_param]
    str = 'cmd /c \"' + cellprofiler + '\" -c -r -p c:\Linda\cellProfiller_pipeline.cppipe -o C:\Linda\\resources\Outputs -i C:\Linda\\resources\Outputs\cropped'
    subprocess.run(str, shell=True)
    print(str)


def crop_and_create_metadata():
    """
    crop_and_create_metadata - crops the image samples using annotations, so that the cropped images contain the labeled leukocytes
    """

    def crop_cell(row):
        """
        crop_cell - crops the image according to given coordinates
        :param row: string (row from file) = the coordinates
        :return: 'error', if the image couldn't be cropped, name of the file otherwise
        """
        # open image
        im = cv2.imread(f"{input_dir}/{row['filename']}")

        # setting the points for cropped image
        x1 = int(row['xmin'])
        x2 = int(row['xmax'])
        y1 = int(row['ymin'])
        y2 = int(row['ymax'])
        im1 = im[y1:y2, x1:x2, :]

        cropped_fname = f"BloodImage_{row['image_id']:04d}_{row['cell_id']:02d}.jpg"

        # save the image
        try:
            cv2.imwrite(f"{output_dir}/{cropped_fname}", im1)
        except:
            return 'error'

        return cropped_fname

    input_dir = arguments_dictionary[sample_dir_param]
    inout_json_dir = '..\generated\\'
    output_dir = arguments_dictionary[sample_dir_param] + '\cropped'

    # create directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    os.chmod(output_dir, 0o777)

    # load metadata
    task1_fp = inout_json_dir + arguments_dictionary[dataset_name_param] + "_metadata.csv"
    task1_df = pd.read_csv(task1_fp)

    # crop each cell and save to file
    task1_df['cell_filename'] = task1_df.apply(crop_cell, axis=1)

    # drop errors
    task1_df = task1_df[task1_df.cell_filename != 'error']

    # save to csv
    task1_df[['cell_filename', 'image_id', 'cell_id', 'cell_type', 'wbc_type']].to_csv(
        inout_json_dir + 'vsetko+original_cropped.csv',
        index=False)


def export():
    """
    export - exports the extraxcted data from the annotations to CSV file
    """
    global filename, wbc_type, xmin, xmax, ymin, ymax
    annotations_json = glob(arguments_dictionary[annotations_path_to_param] + '/*.json')
    output_csv_dir = '..\generated\\'
    df = []
    image_id = 0

    # create the directory
    if not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir)
    else:
        shutil.rmtree(output_csv_dir)
        os.makedirs(output_csv_dir)
    os.chmod(output_csv_dir, 0o777)

    # read the annotations by one
    for file in annotations_json:

        filename = file.split('\\')[-1]
        filename = filename.split('.')[0] + '.jpg'
        row = []

        with open(file, 'r') as f:
            parsed_json = json.load(f)

        # extract the necessary inf from annotations
        for cell_id in range(0, int(parsed_json['Cell Numbers'])):
            cell_json = parsed_json['Cell_' + str(cell_id)]
            blood_cells = 'WBC'
            wbc_type = (cell_json['Label2'])
            xmin = int(cell_json['x1'])
            xmax = int(cell_json['x2'])
            ymin = int(cell_json['y1'])
            ymax = int(cell_json['y2'])

            # add the info to matrix
            row = [filename, image_id, cell_id, blood_cells, wbc_type, xmin, xmax, ymin, ymax]
            df.append(row)

        image_id += 1

    # create dataframe and save it to csv
    data = pd.DataFrame(df, columns=['filename', 'image_id', 'cell_id', 'cell_type', 'wbc_type', 'xmin', 'xmax', 'ymin',
                                     'ymax'])
    data[['filename', 'image_id', 'cell_id', 'cell_type', 'wbc_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv(
        output_csv_dir + arguments_dictionary[dataset_name_param] + '_metadata.csv', index=False)


def select_anotations():
    """
    select_anotations - selects the corresponding annotations from all the annotations to match the chosen sample of images
    """
    input_json_dir = arguments_dictionary[annotations_path_from_param]
    output_json_dir = arguments_dictionary[annotations_path_to_param]
    intput_image_dir = arguments_dictionary[sample_dir_param]
    jsons = os.listdir(input_json_dir)
    images = os.listdir(intput_image_dir)

    # create the directory
    if not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir)
    else:
        shutil.rmtree(output_json_dir)
        os.makedirs(output_json_dir)
    os.chmod(output_json_dir, 0o777)

    # find and copy the annotation file for all the images
    for name in images:
        name = name.split('.')[0] + '.json'

        if name in jsons:
            shutil.copy2(input_json_dir + '/' + name, output_json_dir)


def select_metadata():
    """
    select_metadata - select necessary information from the metadata created by CellProfiler and save it to CSV file
    """

    def create_connection(db_file):
        """
        create_connection - create a database connection to the SQLite database specified by the db_file
        :param db_file: database file
        :return: the connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)

        return conn

    def select_all_tasks(conn):
        """
        select_all_tasks - query all rows in the tasks table
        :param conn: the connection object
        """

        with open('select_cellprofiler_pipeline.sql', 'r') as sql_file:
            sql_script = sql_file.read()

        # to export as csv file
        with open("../generated/exportDB.csv", "wb") as write_file:
            cursor = conn.cursor()
            cursor.execute(sql_script)
            names = [description[0] for description in cursor.description]
            st = ','.join(names)
            st = st + "\n"
            write_file.write(st.encode())
            rows = cursor.fetchall()
            for row in rows:
                writeRow = ",".join([str(i) for i in row])
                writeRow = writeRow + "\n"
                write_file.write(writeRow.encode())

    conn = create_connection(database)

    with conn:
        select_all_tasks(conn)


def replace_classes(data, target):
    """
    replace_classes - replaces the types of leukocytes with the abstracted class for binary classification (granulocyte/agranulocyte)
    :param data: the dataframe with the data
    :param target: the name of the class in the dataframe, to be replaced
    :return updated data, None in case of failure
    """

    # read the classes from config arguments
    array_real_classes = arguments_dictionary[real_cell_classes_param].split(',')
    array_new_classes = arguments_dictionary[replaced_cell_classes_param].split(',')

    # replace the classes
    if len(array_real_classes) == len(array_new_classes):
        for i in range(len(array_real_classes)):
            data[target] = data[target].replace(array_real_classes[i], array_new_classes[i])

        return data[data[target] != to_remove]
    else:
        print("Couldn't replace classes, classes in config file don't match")
        return None


def active_learning_test(run_count, epoch, base_size, target, path, train_frac, train_random, kBest_k,
                         train_test_size, dataset_name, max_iter_param):
    """
    active_learning_test - performs active learning experiment on the logistic regression model and the given data
    :param run_count: number of replications
    :param epoch: the max size of the training set
    :param base_size: the base size of the training set
    :param target: the name of the predicted class
    :param path: path of the csv file with the extracted features
    :param train_frac: fraction size of the training set
    :param train_random: the random state number
    :param kBest_k: the number of best classes to be chosen (feature selection)
    :param train_test_size: fraction size of splitting the training set
    :param dataset_name: the name of the dataset used
    :param max_iter_param: the number of maximum iterations for logistic regression solver
    """

    def al_test_pred_Y_fill(granulocytes_test_count, feature_selection_test, learner_1, count_all):
        """
        al_test_pred_Y_fill - prediction of the y class with probability
        :param granulocytes_test_count: count of the samples of class 1 on the dataset
        :param feature_selection_test: the set of dependent variables (X)
        :param learner_1: the active learner used
        :param count_all: the count of all the samples
        :return: array of the predictions
        """
        test_pred_Y = (learner_1.predict_proba(feature_selection_test)[:,
                       1] >= granulocytes_test_count / count_all).astype(int)
        return test_pred_Y

    def compare_change(input_list, list_pred, list_real):
        """
        compare_change - compares within the input if the predicted class matches the real one
        :param input_list: the matrix of the input data
        :param list_pred: predicted classes
        :param list_real: labeled classes
        """
        match_11 = sum(1 for a, b in zip(list_pred, list_real) if a == 1 and b == 1)
        match_00 = sum(1 for a, b in zip(list_pred, list_real) if a == 0 and b == 0)
        change_01 = sum(1 for a, b in zip(list_pred, list_real) if a == 0 and b == 1)
        change_10 = sum(1 for a, b in zip(list_pred, list_real) if a == 1 and b == 0)

        input_list = input_list.append([match_00, change_01, change_10, match_11])

    # initializations
    df_st1 = pd.DataFrame(columns=range(epoch))
    df_st1_precision = pd.DataFrame(columns=range(epoch))
    matches_mean = []
    matrix_data_test = []
    matrix_data_train = []

    #read the csv and drop not important classes
    data = pd.read_csv(path)
    data = data.drop(['Image_Metadata_image_id', 'Image_Metadata_cell_id'], axis=1)
    data = replace_classes(data, target)

    # do all the replications
    for j in range(0, run_count):

        print('Replication' + j)
        # split the sample for training and testing
        data_train = data.sample(frac=train_frac, random_state=train_random)
        data_test = data.drop(data_train.index)

        # read the replaced classes and substitute them with numbers (0,1) for binary classification
        classname_array = str(arguments_dictionary[replaced_cell_classes_param]).split(',')
        set_replaced_classes = set(classname_array)
        list_classes = list(set_replaced_classes)

        if to_remove in list_classes:
            list_classes.remove(to_remove)

        list_classes.sort()
        data_train[target] = data_train[target].replace(list_classes[0], 0)
        data_train[target] = data_train[target].replace(list_classes[1], 1)

        # feature selection - select k best features using chi2
        label_encoder = preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(data_train[target])
        X = data_train.iloc[:, 2:]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        feature_selection = SelectKBest(chi2, k=kBest_k).fit_transform(X_scaled, y)

        # split the selected data to train/test for training the model
        X_train, X_test, y_train, y_test = train_test_split(feature_selection, y, test_size=train_test_size,
                                                            random_state=42)
        granulocytes_count, agranulocytes_count = data_train[target].value_counts()

        # create Active Learner - Logistic Regression
        learner_1 = ActiveLearner(
            estimator=LogisticRegression(solver='saga', max_iter=max_iter_param),
            query_strategy=uncertainty_sampling,
            class_weight={0: agranulocytes_count, 1: granulocytes_count}
        )

        # initialize the base sets for the model training
        X_train_base_st1 = X_train[:base_size]
        y_train_base_st1 = y_train[:base_size]
        X_train_new_st1 = X_train[base_size:]
        y_train_new_st1 = y_train[base_size:]
        st1_scores = []
        st1_scores_precision = []
        range_epoch = epoch

        # perform the training for all the defined base set sizes
        for i in range(range_epoch):

            # train the model on the base dataset
            learner_1.fit(X_train_base_st1, y_train_base_st1)
            sum_weight = agranulocytes_count + granulocytes_count

            # predict the Y class
            st1_pred = al_test_pred_Y_fill(granulocytes_count, X_test, learner_1, sum_weight)

            # calculate and append scores achieved
            st1_scores_precision.append(average_precision_score(st1_pred, y_test))
            weights_by_class = [1 if y == 1 else ((granulocytes_count / agranulocytes_count)) for y in y_test]
            st1_scores.append(balanced_accuracy_score(st1_pred, y_test, sample_weight=weights_by_class))

            # pick next sample in the random strategy
            query_idx1, query_sample1 = learner_1.query(X_train_new_st1)

            # add by index to the base set for training
            X_train_base_st1 = np.append(X_train_base_st1, X_train_new_st1[query_idx1], axis=0)
            y_train_base_st1 = np.concatenate([y_train_base_st1, y_train_new_st1[query_idx1]], axis=0)
            X_train_new_st1 = np.concatenate([X_train_new_st1[:query_idx1[0]], X_train_new_st1[query_idx1[0] + 1:]],
                                             axis=0)
            y_train_new_st1 = np.concatenate([y_train_new_st1[:query_idx1[0]], y_train_new_st1[query_idx1[0] + 1:]],
                                             axis=0)

            if i == range_epoch - 2:
                break

        # append the scores to the dataframe
        df_st1 = df_st1.append(pd.Series(st1_scores, index=df_st1.columns), ignore_index=True)
        df_st1_precision = df_st1_precision.append(pd.Series(st1_scores_precision, index=df_st1_precision.columns),
                                                   ignore_index=True)

        # plot the results for the training of the model for the given replication
        plt.plot(list(range(range_epoch)), st1_scores, label='Presnosť modelu pri trénovaní replikácie č.' + str(j))
        plt.xlabel('Veľkosť trénovacej množiny')
        plt.ylabel('Presnosť modelu')
        plt.legend()
        plot_save_name = arguments_dictionary[output_dir_param] + '/plots/' + dataset_name + "_balanced_accuracy_score_" + str(j) + ".png"
        plt.savefig(plot_save_name, bbox_inches='tight')
        plt.show()

        # testing section
        # prepare the tesing data in the same way the training data were prepared
        label_encoder = preprocessing.LabelEncoder()
        y_testing = label_encoder.fit_transform(data_test[target])
        X_testing = data_test.iloc[:, 2:]
        scaler = MinMaxScaler()
        X_testing_scaled = scaler.fit_transform(X_testing)
        feature_selection_test = SelectKBest(chi2, k=kBest_k).fit_transform(X_testing_scaled, y_testing)
        granulocytes_test_count, agranulocytes_test_count = data_test[target].value_counts()
        sum_weight_test = agranulocytes_test_count + granulocytes_test_count

        # predict the Y class for the testing set
        test_pred_Y = al_test_pred_Y_fill(granulocytes_test_count, feature_selection_test, learner_1,
                                          sum_weight_test)

        # evaluate and append the accuracy score of the predictions
        array_weight = []
        array_weight.append([agranulocytes_test_count, granulocytes_test_count])
        weights_by_class_test = [1 if y_t == 1 else ((agranulocytes_test_count / granulocytes_test_count)) for y_t in
                                 y_testing]
        matches_mean.append(balanced_accuracy_score(test_pred_Y, y_testing, sample_weight=weights_by_class_test))
        compare_change(matrix_data_train, y_test, st1_pred)
        compare_change(matrix_data_test, y_testing, test_pred_Y)

        #  csv test y_testing, replace 0,1 with the classes defined in config file
        csv_data_WBC_type = [str.replace(str(w), '0', list_classes[0]) for w in test_pred_Y]
        csv_data_WBC_type = [str.replace(str(w), '1', list_classes[1]) for w in csv_data_WBC_type]

        # save the predictions to CSV file
        csv_data_df = data_test.iloc[:, [0, 1]].copy()
        csv_data_df['Prediction'] = np.array(csv_data_WBC_type)
        csv_data_df.to_csv(arguments_dictionary[output_dir_param] + '/csv/' +
                           arguments_dictionary[test_predictions_filename_param] + str(j) + '.csv', ";")

    # save the average accuracy scores to csv file
    st1_mean = list(df_st1.mean())
    df_st1.append(st1_mean)
    df_st1.to_csv(arguments_dictionary[output_dir_param] + '/csv/' + dataset_name + '_balanced_accuracy_score' + str(run_count) + '.csv', ";", decimal=",")
    file_object = open(arguments_dictionary[output_dir_param] + '/csv/' + dataset_name + '_balanced_accuracy_score' + str(run_count) + '.csv', 'a')
    converted_list = [str(element) for element in st1_mean]
    list_new = ['priemer'] + converted_list
    resultString = ';'.join(list_new)
    resultString = resultString.replace(".", ",")
    file_object.write(resultString)
    file_object.close()

    # print the mean model accuracy for all the replications and plot and save the chart
    print("Presnosť natrénovaného modelu: " + str(st1_mean[range_epoch - 2]))
    plt.plot(list(range(range_epoch)), st1_mean, label='Presnosť modelu pri trénovaní')
    plt.xlabel('Veľkosť trénovacej množiny')
    plt.ylabel('Presnosť modelu')
    plt.legend()
    plot_save_name = arguments_dictionary[output_dir_param] + '/plots/' + dataset_name + "_balanced_accuracy_score_mean.png"
    plt.savefig(plot_save_name, bbox_inches='tight')
    plt.show()

    # print the mean model accuracy on testing set for all the replications and plot and save the chart
    print("Presnosť pri testovaní: " + str(mean(matches_mean)))
    plt.plot(list(range(0, run_count)), matches_mean, label='Presnosť natrenovaného modelu')
    plt.plot(list(range(0, run_count)), [mean(matches_mean)] * run_count, label='Priemerná presnosť')
    plt.xlabel('Počet behov')
    plt.ylabel('Priemerná presnosť pri testovacej množine')
    plt.legend()
    plot_save_name = arguments_dictionary[output_dir_param] + '/plots/' + dataset_name + "_balanced_accuracy_score_test.png"
    plt.savefig(plot_save_name, bbox_inches='tight')
    plt.show()

    # create and plot the mean confusion matrix for training and testing set
    df_matrix_train = pd.DataFrame(matrix_data_train)
    df_matrix_test = pd.DataFrame(matrix_data_test)
    df_matrix_train_mean = list(df_matrix_train.mean())
    df_matrix_test_mean = list(df_matrix_test.mean())
    pole_train = [[df_matrix_train_mean[0], df_matrix_train_mean[1]],
                  [df_matrix_train_mean[2], df_matrix_train_mean[3]]]
    pole_test = [[df_matrix_test_mean[0], df_matrix_test_mean[1]],
                 [df_matrix_test_mean[2], df_matrix_test_mean[3]]]

    # training confusion matrix
    labels = [list_classes[0], list_classes[1]]
    f = seaborn.heatmap(pole_train, annot=True, xticklabels=labels, yticklabels=labels, fmt='f')
    plt.title("Matica chybovosti trénovacej množiny")
    plot_save_name = arguments_dictionary[output_dir_param] + '/plots/' + dataset_name + "_conf_matrix_mean_train.png"
    plt.savefig(plot_save_name, bbox_inches='tight')
    plt.show()

    # testing confusion matrix
    labels = [list_classes[0], list_classes[1]]
    f = seaborn.heatmap(pole_test, annot=True, xticklabels=labels, yticklabels=labels, fmt='f')
    plt.title("Matica chybovosti testovacej množiny")
    plot_save_name = arguments_dictionary[output_dir_param] + '/plots/' + dataset_name + "_conf_matrix_mean_test.png"
    plt.savefig(plot_save_name, bbox_inches='tight')
    plt.show()


# main code
if __name__ == "__main__":

    # global definitions
    global arguments_dictionary, to_remove, database
    global app_dir_param, sample_dir_param, output_dir_param, annotations_path_from_param, annotations_path_to_param, \
        real_cell_classes_param, replaced_cell_classes_param, run_count_param, \
        epoch_param, base_size_param, train_frac_param, train_random_state_param, k_best_classes_param, \
        predicted_class_param, dataset_name_param, test_predictions_filename_param, max_iter_param, \
        train_test_size_param, export_csv_default_name_param, select_annotations_param, crop_and_create_metadata_param, \
        run_cellprofiller_param, select_metadata_from_db_param, perform_active_learning_param

    # initialization
    to_remove = 'None'
    yes_param_value = 'y'

    arguments_dictionary = {}
    app_dir_param = 'app_dir_param'
    sample_dir_param = 'sample_dir_param'
    output_dir_param = 'output_dir_param'
    annotations_path_from_param = 'annotations_path_from_param'
    annotations_path_to_param = 'annotations_path_to_param'
    annotations_path_to_param = 'annotations_path_to_param'
    real_cell_classes_param = 'real_cell_classes_param'
    replaced_cell_classes_param = 'replaced_cell_classes_param'
    run_count_param = 'run_count_param'
    epoch_param = 'epoch_param'
    base_size_param = 'base_size_param'
    train_frac_param = 'train_frac_param'
    train_random_state_param = 'train_random_state_param'
    k_best_classes_param = 'k_best_classes_param'
    predicted_class_param = 'predicted_class_param'
    dataset_name_param = 'dataset_name_param'
    test_predictions_filename_param = 'test_predictions_filename_param'
    max_iter_param = 'max_iter_param'
    train_test_size_param = 'train_test_size_param'
    export_csv_default_name_param = 'export_csv_default_name_param'
    select_annotations_param = 'select_annotations_param'
    crop_and_create_metadata_param = 'crop_and_create_metadata_param'
    run_cellprofiller_param = 'run_cellprofiller_param'
    select_metadata_from_db_param = 'select_metadata_from_db_param'
    perform_active_learning_param = 'perform_active_learning_param'

    # paths
    database = "../generated/DefaultDB.db"
    export_csv_path = '../generated/'
    export_csv_default_path = '../default/'
    export_path_to_use = ''

    # read the config file
    read_config('../resources/config/python_config.txt')

    # decide if the default CSV file with feature extraction is used or the generated shall be used
    if arguments_dictionary.get(export_csv_default_name_param, None) is None:
        export_path_to_use = export_csv_path + 'exportDB.csv'
    else:
        export_path_to_use = export_csv_default_path + arguments_dictionary[export_csv_default_name_param]

    # select annotations if required
    if arguments_dictionary[select_annotations_param] is yes_param_value:
        select_anotations()
        print("1. Selecting annotations successful")

    # crop and create metadata if required
    if arguments_dictionary[select_annotations_param] is yes_param_value and \
            arguments_dictionary[crop_and_create_metadata_param] is yes_param_value:
        export()
        crop_and_create_metadata()
        print("2. Cropping images and creating metadata successful")

    # run cellprofiller if required
    if arguments_dictionary[select_annotations_param] is yes_param_value and \
            arguments_dictionary[crop_and_create_metadata_param] is yes_param_value and \
            arguments_dictionary[run_cellprofiller_param] is yes_param_value:
        run_cellprofiler()
        print("3. Running CellProfiler successful")

    # select metadata from CellProfiler if required
    if arguments_dictionary[select_metadata_from_db_param] is yes_param_value and os.path.exists(database):
        select_metadata()
        print("4. Selecting metadata successful")

    # perform active learning if required
    if arguments_dictionary[perform_active_learning_param] is yes_param_value :
        active_learning_test(int(arguments_dictionary[run_count_param]),
                             int(arguments_dictionary[epoch_param]),
                             int(arguments_dictionary[base_size_param]),
                             arguments_dictionary[predicted_class_param],
                             export_path_to_use,
                             float(arguments_dictionary[train_frac_param]),
                             int(arguments_dictionary[train_random_state_param]),
                             int(arguments_dictionary[k_best_classes_param]),
                             float(arguments_dictionary[train_test_size_param]),
                             arguments_dictionary[dataset_name_param],
                             int(arguments_dictionary[max_iter_param]))  # exp1
        print("5. Running active learning model successful")
