import gradio as gr
import pandas as pd
import cutie
import os
import static_frame as sf
from termcolor import colored
import numpy as np
import cutie
import matplotlib.pyplot as plt
import seaborn as sns
import dtale
import plotly.express as px
from System.Auxiliary import auxiliary_fun
# from System.Modules import Dacquisition
from System.Modules import DprepNcleaning
from System.Modules import eda
from System.Modules import Mbuilding

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

    ###casteo 
    input_casted = auxiliary_fun.cast_input(data_input, rosseta)
    input_casted = input_casted[input_casted.columns[::-1]]
    input_casted = input_casted.to_numpy().astype(np.float64)
    ###formato
    predicted_val = selected_model.predict(input_casted[0].reshape(1,-1)) 
    predicted_val = auxiliary_fun.de_cast_PREDICTION(pd.DataFrame(predicted_val), [TARGET_COLUMN], rosseta)
    predicted_val = predicted_val.iloc[0,0] 
    ret = 'Prediction: '+str(predicted_val)
    if TARGET_TYPE == 'boolean' or TARGET_TYPE == 'classes':
        predicted_val_proba = selected_model.predict_proba(input_casted[0].reshape(1,-1))
        ret = ret + ', Propability: '+str(predicted_val_proba[0][predicted_val.astype(int)])

    return ret



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

                    gr.Markdown("## The dataset is: ")
                    gr.DataFrame(DATA, interactive='False')

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
                            if TARGET_TYPE == 'continuous': #print depending of target type
                                text_output_1 = '\nThe target column type is '
                                text_output_2 = ', the elements range is:'
                                text_output_3 = '('+str(DATA[TARGET_COLUMN].min())+', '+str(DATA[TARGET_COLUMN].max())+')'
                            else: #discrete case
                                text_output_1 = '\nThe target column type is ' + 'discrete ('                                 
                                text_output_2 = ')'+', the classes are:'
                                text_output_3 = '\n'
                                for unique_element in DATA[TARGET_COLUMN].unique(): 
                                    print(colored(str(unique_element)+' ', 'green', attrs=['bold']), end='')
                                    text_output_3 = text_output_3+str(unique_element)+' - '
                                text_output_3 = text_output_3[:len(text_output_3)-3]
                        with gr.Row():
                            with gr.Column():
                                gr.Label(text_output_1+TARGET_TYPE.upper()+text_output_2, show_label=False)
                                gr.Label(text_output_3, show_label=False)
                            fig_target = plt.figure()
                            # sns.histplot(data=DATA[TARGET_COLUMN], kde=True, stat='percent')
                            # plt.title('Histogram of distribution of target variable')
                            px.bar(DATA, x=TARGET_COLUMN)
                            gr.Plot(fig_target, show_label=False, scale = 1)


                        #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
                        global rosseta
                        DATA_casted, rosseta = auxiliary_fun.d_cast(DATA, TARGET_COLUMN)
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

                    with gr.Row():
                        with gr.Column():
                            gr.Label(text_output_1+TARGET_TYPE.upper()+text_output_2, show_label=False)
                            gr.Label(text_output_3, show_label=False)
                        fig_target = plt.figure()
                        sns.histplot(data=DATA[TARGET_COLUMN], kde=True, stat='percent', discrete=True)
                        gr.Plot(fig_target, show_label=False, scale = 1)
                        
                    #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
                    global rosseta
                    DATA_casted, rosseta = auxiliary_fun.d_cast(DATA, TARGET_COLUMN)
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

                    #aca tengo q descastear y mandar a eda
##########################    ##########################    ##########################    ##########################    ##########################    ##########################


        with gr.Tab(label="Exploratory data analysis"):

            EDA_RFLAG = gr.State(0)            
            calculate_button = gr.Button("Start exploratory data analysis (EDA)")
            calculate_button.click(lambda count: count + 1, EDA_RFLAG, EDA_RFLAG, scroll_to_output=True)

            @gr.render(inputs=[EDA_RFLAG])
            def scewed_data_cast(EDA):
                if EDA>0:
                    gr.Markdown("## Table with descriptive statistics of variables: ")
                    gr.DataFrame(DATA.describe(), interactive='False')
                    gr.Markdown("## Table with information of the variables: ")
                    gr.DataFrame(manualEDA, interactive='False')
                    gr.Markdown("## Table with information of the rows: ")
                    gr.DataFrame(missing4rows, interactive='False')

                    ### Step 3.2: Exploratory Data Analyzis (AUTO)###
                    dtale.show(DATA) 
                    dtale.show(open_browser=True)
                    # dtale.show()

##########################    ##########################    ##########################    ##########################    ##########################    ##########################
  

        with gr.Tab(label="Data integrity"):
            INT_RFLAG = gr.State(0)            
            calculate_button = gr.Button("Start exploratory data integrity")
            calculate_button.click(lambda count: count + 1, INT_RFLAG, INT_RFLAG, scroll_to_output=True)

            @gr.render(inputs=[INT_RFLAG])
            def scewed_data_cast(INT):
                if INT>0:
                    
                    gr.Markdown('## Resultado de Data Cleaning:')
                    gr.Markdown('Cantidad de columnas eliminadas: '+
                                str(len(DROP_COL))+
                                ' of '+
                                str(len(DATA_cleaned.columns))+
                                ' ('+
                                str( round(len(DROP_COL)/len(DATA_cleaned.columns)*100, 2) )+
                                '%)')
                    gr.Markdown('The result of the number of patients is: '+str(len(DATA_cleaned)))


                    DATA_cleaned_decasted = auxiliary_fun.de_cast_PREDICTION(DATA_cleaned, DATA_cleaned.columns, rosseta)  
                    gr.DataFrame(DATA, interactive='False')
                    gr.DataFrame(DATA_cleaned_decasted, interactive='False')
                    ### Step 3.2: Exploratory Data Analyzis (AUTO)###
                    # dtale.show(DATA_cleaned_decasted) 
                    # dtale.show(open_browser=True)
                    # dtale.show()


##########################    ##########################    ##########################    ##########################    ##########################    ##########################

        with gr.Tab(label="Automated machine learning"):

            MODEL_RESULT_RFLAG = gr.State(0)
            calculate_button = gr.Button("Start model shake")
            calculate_button.click(lambda count: count + 1, MODEL_RESULT_RFLAG, MODEL_RESULT_RFLAG, scroll_to_output=True)

            @gr.render(inputs=[MODEL_RESULT_RFLAG])
            def scewed_data_cast(MODEL_RESULT):
                if MODEL_RESULT>0:

                        ### Step 5: Model Building       
                    global model_list
                    global model_info
                    model_info, model_list, figure_features, fig_ROC, disp = Mbuilding.model_shake(DATA_cleaned, TARGET_COLUMN, TARGET_TYPE, Fast = False)                    
                    gr.DataFrame(model_info.drop(['Target column', 'Target type'], axis=1).dropna(axis='columns'), label="Table with information of the trained models:", scale=1, interactive='False')

                    with gr.Row():
                        gr.Plot(figure_features, show_label=False, scale = 1)
                        if TARGET_TYPE == 'boolean':
                            gr.Plot(fig_ROC, show_label=False)
                            gr.Plot(disp, show_label=False)

                    return_model = sns.lmplot(data=model_info.rename(columns={'Normalization method':'Norm.', 'Feature selection method':'Feat.'}), 
                                              x="Cross-validation ID", 
                                              y="Score", 
                                              row="Norm.", 
                                              col="Feat.", 
                                              hue='Model name',
                                              palette="crest", 
                                              ci=None,height=4, 
                                              scatter_kws={"s": 50, "alpha": 1}) 
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
                    sorted_models = model_info.sort_values(by=['AUC','Score','F1 score','Brier score loss'], ascending=False)[0:10][['Model name','AUC','Score','F1 score','Brier score loss']].round(2) 
                    sorted_models = sorted_models.dropna(axis='columns') 
                    global model_selection_list
                    global idx_model_sl_lst
                    model_selection_list = []
                    idx_model_sl_lst = []
                    for row in sorted_models.iterrows():
                        model_atributes = ''
                        for idx, column in enumerate(row[1]):
                            model_atributes = model_atributes+row[1].axes[0][idx]+': '+str(column)+', '
                            
                        model_atributes = model_atributes[:len(model_atributes)-2] 
                        model_selection_list.append(model_atributes)
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
                    feature_used =  model_info['Features used'].loc[index_model]  
                    training_example = DATA_cleaned[feature_used].reset_index()
                    training_example = training_example.drop("index", axis='columns')
                    prediction_example_casted = selected_model.predict(training_example.to_numpy())
                    prediction_example_casted = pd.DataFrame(prediction_example_casted, columns=['Prediction'])
                    if TARGET_TYPE == 'boolean' or TARGET_TYPE == 'classes':
                        prediction_example_proba = selected_model.predict_proba(training_example.to_numpy())

                    #real value ###robablemente tenga q descastear esto tambien , mejor usar DATA_cleaned_decasted
                    real_val =  DATA_cleaned[TARGET_COLUMN].reset_index()
                    real_val = real_val.drop("index", axis='columns')
                    real_val = real_val.rename(columns={TARGET_COLUMN:'Real value'})


                    #DeCast training data, prediction and real_value
                    training_example = auxiliary_fun.de_cast_PREDICTION(training_example, feature_used, rosseta)
                    prediction_example = auxiliary_fun.de_cast_PREDICTION(prediction_example_casted, [TARGET_COLUMN], rosseta)
                    real_val = auxiliary_fun.de_cast_PREDICTION(real_val, [TARGET_COLUMN], rosseta) 
                    
                    indexes_unique = []
                    for unique_prediction in prediction_example['Prediction'].unique():
                        aux_indexes = prediction_example[prediction_example['Prediction']==unique_prediction].index
                        indexes_unique.append(aux_indexes[0])
                    with gr.Row():
                        prediction_example_DF = pd.concat([training_example[training_example.columns[::-1]].loc[indexes_unique].head(5), real_val.loc[indexes_unique].head(5),prediction_example.loc[indexes_unique].head(5)],axis=1)
                        if TARGET_TYPE == 'boolean' or TARGET_TYPE == 'classes':
                            class_pred_proba = []
                            count = 0
                            for idx in prediction_example_casted.loc[indexes_unique].astype(int).iterrows():
                                class_pred_proba.append(round(prediction_example_proba[indexes_unique[count]][idx[1]['Prediction']],2))
                                count = count+1

                            pep_DF = pd.DataFrame({'Probability':class_pred_proba}) 
                            prediction_example_DF = pd.concat([prediction_example_DF.reset_index(drop=True),pep_DF.head(5)],axis=1) 

                        gr.DataFrame(prediction_example_DF, label="Example of inputs:", scale=5, interactive='False')

                    
                    gr.Interface(
                        fn=get_entry,
                        inputs=[gr.Dataframe(
                                headers=list(model_info['Features used'].loc[index_model][::-1]),
                                row_count=1,
                                col_count=(len(model_info['Features used'].loc[index_model]), "fixed"),
                                interactive=True,
                                scale = 5
                                )],
                        outputs="textbox",
                        description="Enter input to predict: ",
                    )

                    
# Launch the Gradio interface
demo.launch(share=True)
# demo.launch()
