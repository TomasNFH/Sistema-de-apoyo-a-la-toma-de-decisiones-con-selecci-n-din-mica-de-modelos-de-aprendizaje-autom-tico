import pycaret
import System as sy

#dynamic system modelation?
ret = sy.dyn_model_selection(FAST = True, PLOT = True)


# # loading sample dataset from pycaret dataset module
# from pycaret.datasets import get_data
# data = get_data('diabetes')

# # import pycaret classification and init setup
# from pycaret.classification import *
# s = setup(data, target = 'Class variable', session_id = 123)

# breakpoint()

# # import ClassificationExperiment and init the class
# from pycaret.classification import ClassificationExperiment
# exp = ClassificationExperiment()

# # check the type of exp
# print(type(exp))

# # init setup on exp
# print(exp.setup(data, target = 'Class variable', session_id = 123))

# # compare baseline models
# best = compare_models()

# # compare models using OOP
# # print(exp.compare_models())

# # plot confusion matrix
# # plot_model(best, plot = 'confusion_matrix')

# # plot AUC
# # plot_model(best, plot = 'auc')

# # plot feature importance
# # plot_model(best, plot = 'feature')


# evaluate_model(best)

# # predict on test set
# holdout_pred = predict_model(best)

# # show predictions df
# holdout_pred.head()


# # copy data and drop Class variable

# new_data = data.copy()
# new_data.drop('Class variable', axis=1, inplace=True)
# new_data.head()

# # predict model on new_data
# predictions = predict_model(best, data = new_data)
# predictions.head()