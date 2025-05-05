from System import Workflow

#dynamic system modelation?
ret = Workflow.dyn_model_selection(FAST = True, PLOT = True)

                    # plt.figure(figsize=(8, 6))
                    # plt.plot(metrics_result['roc_curve'][0], metrics_result['roc_curve'][1], color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics_result["roc_curve"][2]:.2f})')
                    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    # plt.xlim([0.0, 1.0])
                    # plt.ylim([0.0, 1.05])
                    # plt.xlabel('False Positive Rate')
                    # plt.ylabel('True Positive Rate')
                    # plt.title(f'{name} - Receiver Operating Characteristic')
                    # plt.legend(loc="lower right")
                    # plt.show()
                    # breakpoint()
                    ###PARAG

# (Pdb) row['False Positive Rate']
# array([0.        , 0.15384615, 0.76923077, 0.76923077, 0.76923077,
#        1.        , 1.        ])
# (Pdb) row['True Positive Rate']  
# array([0.        , 0.        , 0.        , 0.28571429, 0.42857143,
#        0.42857143, 1.        ])
