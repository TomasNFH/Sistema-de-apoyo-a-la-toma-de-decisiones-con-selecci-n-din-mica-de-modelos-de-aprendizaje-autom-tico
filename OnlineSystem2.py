import gradio as gr
import pandas as pd
import cutie
import os
import auxiliary_fun
import static_frame as sf
from termcolor import colored
import numpy as np
import cutie
import matplotlib.pyplot as plt
import seaborn as sns
import DprepNcleaning
import eda
import Mbuilding
import dtale

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##########################    ##########################    ##########################    ##########################    ##########################    ##########################
##########################    ##########################    ##########################    ##########################    ##########################    ##########################


def var_acquisition(column_name):
    global DATA
    global CHECK #sacar de aca
    global TARGET_TYPE
    global drop_target_col_values #needed for the scewed cast
    global TARGET_COLUMN #needed for data drop in scewed data
    TARGET_TYPE = 'Not identify'
    TARGET_COLUMN = column_name

    #check if the user already selected something
    if len(TARGET_COLUMN) != 0: #check if the
        
        ### 1 drop rows from data by NaN in predict column
        idx_drop_from_predict = DATA.index[DATA[TARGET_COLUMN].isna() == True].tolist()
        DATA = DATA.drop(idx_drop_from_predict)

        ### 3 chech predicted column type
        # TARGET_TYPE = 'Not identify'
        unique_values = DATA[TARGET_COLUMN].unique()
        if len(unique_values) == 2: #if we have only two type of values it means is boolean
            TARGET_TYPE = 'boolean'

        ### 3.1 verify scewed predict data (to cast to boolean) <WE DO IT IN RENDER>

        ### 3.2 Coff of unique check  
        len_target_col = len(DATA[TARGET_COLUMN]) 
        R = len(unique_values)/len_target_col #coeficient of number of unique in relation to number of rows
        number_of_classes = 10

        #if we have to many classes in relation of samples, ask to group the classes 
        #sceewed check is added becouse we dont want to over write what we done
        if R>(number_of_classes/100):    
            if len(unique_values)!=2: #dont enter if we already classify as boolean
                TARGET_TYPE = 'continuous'   
        if R<=(number_of_classes/100): 
            if len(unique_values)!=2: #dont enter if we already classify as boolean
                TARGET_TYPE = 'classes'    
    return 0

def scewed_selection(selection):
    global DATA
    global TARGET_TYPE
    global TARGET_COLUMN
    global drop_target_col_values
    if selection == "Yes":
        TARGET_TYPE = 'boolean'
        for drop_val in drop_target_col_values:
            drop_row = DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==drop_val].index
            DATA = DATA.drop(drop_row)
    return 0

def get_column_model(number):
    global current_model_selected
    current_model_selected = number
    return 0

def get_entry(data_input):
    # breakpoint()
    columns = data_input.columns
    data_input = data_input.to_numpy()  

    ###casteo
    input_casted = auxiliary_fun.cast_input(data_input, columns, rosseta)


    ###formato
    breakpoint()
    predicted_val = selected_model.predict(data_input.reshape(1,-1))
    return predicted_val[0]


##########################    ##########################    ##########################    ##########################    ##########################    ##########################
##########################    ##########################    ##########################    ##########################    ##########################    ##########################

css = """
.custom-button {
    background-color: #4CAF50; /* Green background */
    color: white;              /* White text */
    border: none;              /* No border */
    padding: 15px 32px;        /* Some padding */
    text-align: center;        /* Centered text */
    text-decoration: none;     /* No underline */
    display: inline-block;     /* Inline-block */
    font-size: 16px;           /* Large font size */
    margin: 4px 2px;           /* Some margin */
    cursor: pointer;           /* Pointer/hand icon */
    border-radius: 8px;        /* Rounded corners */
}
"""

class Tomas_Color:
    all = []

    def __init__(
        self,
        c50: str,
        c100: str,
        c200: str,
        c300: str,
        c400: str,
        c500: str,
        c600: str,
        c700: str,
        c800: str,
        c900: str,
        c950: str,
        name: str | None = None,
    ):
        self.c50 = c50
        self.c100 = c100
        self.c200 = c200
        self.c300 = c300
        self.c400 = c400
        self.c500 = c500
        self.c600 = c600
        self.c700 = c700
        self.c800 = c800
        self.c900 = c900
        self.c950 = c950
        self.name = name
        Tomas_Color.all.append(self)

    def expand(self) -> list[str]:
        return [
            self.c50,
            self.c100,
            self.c200,
            self.c300,
            self.c400,
            self.c500,
            self.c600,
            self.c700,
            self.c800,
            self.c900,
            self.c950,
        ]  
ing_bio_green = Tomas_Color(
    name="ing_bio_green",
    c50="#C0E3DD",
    c100="#C0E3DD",
    c200="#309284",
    c300="#309284",
    c400="#309284",
    c500="#309284",
    c600="#309284",
    c700="#19575C",
    c800="#309284",
    c900="#09B474",
    c950="#309284",
)
    

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=ing_bio_green,secondary_hue="blue",neutral_hue= 'gray',text_size='sm', spacing_size='sm', radius_size='sm')) as demo:
    with gr.Tabs():
        DATAcasted = 'Not uploaded'

##########################    ##########################    ##########################    ##########################    ##########################    ##########################

        with gr.Tab(label="Data acquisition and cleaning"):

            # define FLAGS for render order
            column_dropdown_RFLAG = gr.State(0)
            show_type_RFLAG = gr.State(0)

            data = gr.File(label="Upload CSV / XLSX file", type="filepath", scale = 5) #data id the file uploaded

            # render:
            #           <data head> DATAFRAME
            #           <User check of auto detection of target column:> CHECK
            @gr.render(inputs=[data])
            def show_split(FILE):

                #enter if a file is loaded (define DATA and its columns Variables_OF_DATA)
                if FILE != None: 
                    global DATA
                    global VARIABLES_OF_DATA
                    global target_check_RFLAG 

                    # global target_check_btn
                    if FILE.name[len(FILE.name)-5:] == '.xlsx':
                        DATA = pd.read_excel(FILE.name)
                    if FILE.name[len(FILE.name)-4:] == '.csv':
                        DATA = pd.read_csv(FILE.name)
                    VARIABLES_OF_DATA = DATA.columns #used in target column acq

                    gr.Markdown("## The head of the dataset is: ")
                    gr.DataFrame(DATA.head(5), interactive='False')

                    column_dropdown = gr.Dropdown(list(VARIABLES_OF_DATA), label="Please select the varaible to predict from the next list: ", filterable=False)
                    column_dropdown.change(fn=var_acquisition, inputs=column_dropdown, outputs=None, scroll_to_output=True)
                    column_dropdown.input(lambda count: count + 1, column_dropdown_RFLAG, column_dropdown_RFLAG, scroll_to_output=True)
                    
            #render: 
            #           <The target type is:> 
            @gr.render(inputs=[column_dropdown_RFLAG])
            def scewed_data_cast(column_dropdown):
                SCEWED_FLAG = False
                if column_dropdown>1:                
                    global drop_target_col_values #for scewed function

                    ### 3.1 verify scewed predict data (to cast to boolean)
                    unique_values = DATA[TARGET_COLUMN].unique()
                    SCEWED_TARGET_COL = pd.DataFrame(columns = ['Value id', 'Value', 'Count', 'Occurences (%)'])
                    len_target_col = len(DATA[TARGET_COLUMN])
                    for k,unique_val in enumerate(unique_values):
                        unique_count = len(DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==unique_val]) 
                        SCEWED_TARGET_COL.loc[len(SCEWED_TARGET_COL.index)] = [k, unique_val, unique_count, (unique_count/len_target_col)*100]
                    drop_target_col_values = SCEWED_TARGET_COL[SCEWED_TARGET_COL['Occurences (%)']<10]['Value'] 
                    target_col_values = SCEWED_TARGET_COL[SCEWED_TARGET_COL['Occurences (%)']>=10]['Value'] 

                    ###In the case that values concentrate in only two values and we have something to drop <unque!=2>, ask if the user want to drop the columns that dont have much samples (making it a boolean) 
                    if len(target_col_values) == 2 and len(unique_values)!=2:
                        SCEWED_FLAG= True

                    #if we dont hace a scewed data -> WE SHOW THE RESULT <SAME AS LINE 173>
                    else: 
                        if TARGET_TYPE != 'Not identify': #take it depending if is boolean or classes
                            if TARGET_TYPE == 'continuous': #print depending of target typr
                                text_output_1 = '\nThe target column type is '
                                text_output_2 = ', the elements are:'
                            else: #discrete case
                                # print(colored('\nThe target column type is ', 'black', attrs=['bold']), end='')
                                text_output_1 = '\nThe target column type is ' + 'discrete ('                                 
                                text_output_2 = ')'+', the classes are:'
                            # if TARGET_TYPE == 'boolean': print(colored(DATA[TARGET_COLUMN].unique(), 'green', attrs=['bold']))
                            text_output_3 = '\n'
                            for unique_element in DATA[TARGET_COLUMN].unique(): 
                                print(colored(str(unique_element)+' ', 'green', attrs=['bold']), end='')
                                text_output_3 = text_output_3+str(unique_element)+' - '

                        with gr.Column():
                            gr.Label(text_output_1+TARGET_TYPE.upper()+text_output_2, show_label=False)
                            gr.Label(text_output_3[:len(text_output_3)-3] , show_label=False)

                        #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
                        global rosseta
                        DATA_casted, rosseta = auxiliary_fun.d_cast(DATA, TARGET_COLUMN)
                        # np.searchsorted(uniqueVAL_target, predicted, side='right') 
                        # DATA_casted = DATA
                        print(colored(DATA_casted, 'yellow'))

                            ### Step 3.1: Exploratory Data Analyzis (MANUAL)###
                        global manualEDA, missing4rows
                        manualEDA, missing4rows = eda.ManualEDAfun(DATA_casted) #data no tiene a predicted column
                        
                            ### Step 2: Data Cleaning ###    
                        #min_porcentage_col if missing>10 for a column, we drop it
                        global DATA_cleaned
                        global DROP_COL
                        global DROP_ROW
                        DATA_cleaned, DROP_COL, DROP_ROW = DprepNcleaning.data_cleaning(DATA_casted, min_porcentage_col = 10, min_porcentage_row = 0)
                        # DATA_cleaned = DATA_casted

                    if SCEWED_FLAG == True:
                        scewed_yes_no_btn = gr.Radio(["Yes", "No"], label="Cast to boolean (once selected there is no undo):")
                        scewed_yes_no_btn.input(fn=scewed_selection, inputs=scewed_yes_no_btn, outputs=None) #here we cast DATA (when the button is press there is not way back)
                        scewed_yes_no_btn.input(lambda count: count + 1, show_type_RFLAG, show_type_RFLAG, scroll_to_output=True)

            # to show the result in the case of scewed data (cast or not cast) <SAME AS LINE 160>
            @gr.render(inputs=[show_type_RFLAG])
            def scewed_data_cast(SHOW_TYPE):
                if SHOW_TYPE>0:
                    text_output_1 = '\nThe target column type is ' + 'discrete ('                                 
                    text_output_2 = ')'+', the classes are:'
                    text_output_3 = '\n'
                    for unique_element in DATA[TARGET_COLUMN].unique(): 
                        print(colored(str(unique_element)+' ', 'green', attrs=['bold']), end='')
                        text_output_3 = text_output_3+str(unique_element)+' - '

                    with gr.Column():
                        gr.Label(text_output_1+TARGET_TYPE.upper()+text_output_2, show_label=False)
                        gr.Label(text_output_3[:len(text_output_3)-3] , show_label=False)
                        
                    #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
                    global rosseta
                    DATA_casted, rosseta = auxiliary_fun.d_cast(DATA, TARGET_COLUMN)
                    # DATA_casted = DATA

                        ### Step 3.1: Exploratory Data Analyzis (MANUAL)###
                    global manualEDA, missing4rows
                    manualEDA, missing4rows = eda.ManualEDAfun(DATA_casted) #data no tiene a predicted column
                    
                        ### Step 2: Data Cleaning ###    
                    #min_porcentage_col if missing>10 for a column, we drop it
                    global DATA_cleaned
                    global DROP_COL
                    global DROP_ROW
                    DATA_cleaned, DROP_COL, DROP_ROW = DprepNcleaning.data_cleaning(DATA_casted, min_porcentage_col = 10, min_porcentage_row = 0)
                    # DATA_cleaned = DATA_casted

##########################    ##########################    ##########################    ##########################    ##########################    ##########################


        with gr.Tab(label="Exploratory data analysis"):

            EDA_RFLAG = gr.State(0)            
            calculate_button = gr.Button("Start exploratory data analysis (EDA)")
            calculate_button.click(lambda count: count + 1, EDA_RFLAG, EDA_RFLAG, scroll_to_output=True)

            @gr.render(inputs=[EDA_RFLAG])
            def scewed_data_cast(EDA):
                if EDA>0:
                    gr.Markdown("## Table with descriptive statistics of variables: ")
                    gr.DataFrame(DATA_cleaned.describe(), interactive='False')
                    gr.Markdown("## Table with information of the variables: ")
                    gr.DataFrame(manualEDA, interactive='False')
                    gr.Markdown("## Table with information of the rows: ")
                    gr.DataFrame(missing4rows, interactive='False')

                    gr.Markdown('## Resultado de Data Cleaning:')
                    gr.Markdown('Cantidad de columnas eliminadas: '+
                                str(len(DROP_COL))+
                                ' of '+
                                str(len(DATA_cleaned.columns))+
                                ' ('+
                                str( round(len(DROP_COL)/len(DATA_cleaned.columns)*100, 2) )+
                                '%)')
                    gr.Markdown('The result of the number of patients is: '+str(len(DATA_cleaned)))

                    ### Step 3.2: Exploratory Data Analyzis (AUTO)###
                    dtale.show(DATA_cleaned) #hacerle decast !!!!! 
                    dtale.show(open_browser=True)
                    dtale.show()

##########################    ##########################    ##########################    ##########################    ##########################    ##########################


        with gr.Tab(label="Dynamic models"):

            MODEL_RESULT_RFLAG = gr.State(0)
            calculate_button = gr.Button("Start model shake")
            calculate_button.click(lambda count: count + 1, MODEL_RESULT_RFLAG, MODEL_RESULT_RFLAG, scroll_to_output=True)

            @gr.render(inputs=[MODEL_RESULT_RFLAG])
            def scewed_data_cast(MODEL_RESULT):
                if MODEL_RESULT>0:

                        ### Step 5: Model Building       
                    print(DATA_cleaned)
                    print(TARGET_COLUMN)
                    print(TARGET_TYPE)
                    global model_list
                    global model_info
                    model_info, model_list, figure_features, fig_ROC, disp = Mbuilding.model_shake(DATA_cleaned, TARGET_COLUMN, TARGET_TYPE, Fast = True)                    
                    gr.DataFrame(model_info, label="Table with information of scores of the models:", scale=1, interactive='False')

                    with gr.Row():
                        gr.Plot(figure_features, show_label=False, scale = 1)
                        if TARGET_TYPE == 'boolean':
                            gr.Plot(fig_ROC, show_label=False)
                            gr.Plot(disp, show_label=False)

                    #change labels to better visualization
                    # Feature_methods = ['Intrinsic method','Filter method','Wrapper method']
                    # Normalization_methods = ['No', 'Min-Max', 'Z-score']
                    # NormMtd_used = len(model_info['Normalization method'].unique())  
                    # FeatSel_used = len(model_info['Feature selection method'].unique()) 
                    # # figure_return, axes_return = plt.subplots(NormMtd_used, FeatSel_used, sharey='row')

                    # figure_return = plt.figure()
                    # fig_model, axes_model = plt.subplots(NormMtd_used, FeatSel_used, sharey='row') 
                    # fig_idx = 0
                    # for current_norm in range(NormMtd_used):
                    #     print(Normalization_methods[current_norm])
                    #     for current_feat in range(FeatSel_used):
                    #         print(Feature_methods[current_feat])
                    #         # fig_idx = current_norm*current_feat

                    #         # plt.subplot(NormMtd_used, FeatSel_used, fig_idx)
                    #         # plt.plot()





                    # model_info_plot = model_info.rename(columns={'Normalization method':'Norm.', 'Feature selection method':'Feat.'}) 
                    
                    return_model = sns.lmplot(data=model_info.rename(columns={'Normalization method':'Norm.', 'Feature selection method':'Feat.'}), 
                                              x="Cross-validation ID", 
                                              y="Score", 
                                              row="Norm.", 
                                              col="Feat.", 
                                              hue='Model name',
                                              palette="crest", 
                                              ci=None,height=4, 
                                              scatter_kws={"s": 50, "alpha": 1}) 
                    # # return_model.legend_.remove()
                    # # plt.legend(title="Model name", loc='upper left', bbox_to_anchor=(1, 1))
                    # # plt.suptitle('test', x=1.2, y=.1)
                    # return_model.text(2, 1.75, "Custom text", 
                    #     fontsize = 12,          # Size
                    #     fontstyle = "oblique",  # Style
                    #     color = "red",          # Color
                    #     ha = "center", # Horizontal alignment
                    #     va = "center") # Vertical alignment 
                    figure_return = return_model.fig  
                    with gr.Row():
                        gr.Plot(figure_return, show_label=False)


##########################    ##########################    ##########################    ##########################    ##########################    ##########################


        with gr.Tab(label="Models predictor"):

            MODEL_START_RFLAG = gr.State(0)
            dropdown_Mpredictor_RFLAG = gr.State(0)
            calculate_button = gr.Button("Start model selection")
            calculate_button.click(lambda count: count + 1, MODEL_START_RFLAG, MODEL_START_RFLAG, scroll_to_output=True)
  
            @gr.render(inputs=[MODEL_START_RFLAG])
            def scewed_data_cast(MODEL_START):
                if MODEL_START>0:
                    top10_by_AUC = model_info.sort_values(by=['AUC','Score','F1 score'], ascending=False)[0:10][['Model name','AUC','Score','F1 score']]
                    global model_selection_list
                    global idx_model_sl_lst
                    model_selection_list = []
                    idx_model_sl_lst = []
                    for row in top10_by_AUC.iterrows():
                        aux = str(row[1]['Model name'])+': <AUC: '+str(round(row[1]['AUC'],3))+', Score:' 
                        aux2 = aux + str(round(row[1]['Score'],3)) + ', F1 score:'+str(round(row[1]['F1 score'],3))+'>'
                        model_selection_list.append(aux2)
                        idx_model_sl_lst.append(row[0])

                    column_dropdown = gr.Dropdown(choices=model_selection_list, filterable=False)

                    # Set the function to be called when the dropdown value changes
                    column_dropdown.input(fn=get_column_model, inputs=column_dropdown, outputs=None, scroll_to_output=True)
                    column_dropdown.input(lambda count: count + 1, dropdown_Mpredictor_RFLAG, dropdown_Mpredictor_RFLAG, scroll_to_output=True)


            @gr.render(inputs=[dropdown_Mpredictor_RFLAG])
            def scewed_data_cast(dropdown_Mpredictor):

                if dropdown_Mpredictor>1:
                    global selected_model
                    current_model_selected_idx = model_selection_list.index(current_model_selected)
                    index_model = idx_model_sl_lst[current_model_selected_idx] 
                    selected_model = model_list[dropdown_Mpredictor] 
                    training_example = DATA_cleaned[model_info['Features used'].loc[index_model]].reset_index()
                    training_example = training_example.drop("index", axis='columns')
                    prediction_example = selected_model.predict(training_example.to_numpy())
                    prediction_example = pd.DataFrame(prediction_example, columns=['Prediction'])
                    print('antes')
                    print(prediction_example)
                    prediction_example = auxiliary_fun.de_cast_PREDICTION(prediction_example, TARGET_COLUMN, rosseta)
                    print('despues')
                    print(prediction_example)
                    print('el model sel es')
                    print(selected_model)
                    indexes_unique = []
                    for unique_prediction in prediction_example['Prediction'].unique():
                        # print(unique_prediction)
                        aux_indexes = prediction_example[prediction_example['Prediction']==unique_prediction].index
                        indexes_unique.append(aux_indexes[0])

                    with gr.Row():
                        # breakpoint()
                        gr.DataFrame(training_example[training_example.columns[::-1]].loc[indexes_unique].head(5)  , label="Example of inputs:", scale=5, interactive='False')
                        gr.DataFrame(prediction_example.loc[indexes_unique].head(5) , label="Prediction:", scale=1, interactive='False')

                    # breakpoint()
                    gr.Interface(
                        fn=get_entry,
                        inputs=[gr.Dataframe(
                                headers=list(model_info['Features used'].loc[index_model][::-1]),
                                # datatype=["str", "number", "str"],
                                row_count=1,
                                col_count=(len(model_info['Features used'].loc[index_model]), "fixed"),
                                interactive=True,
                                scale = 5
                                )],
                        outputs="textbox",
                        description="Enter input to predict: ",
                    )

                    
# Launch the Gradio interface
# demo.launch(share=True)
demo.launch()