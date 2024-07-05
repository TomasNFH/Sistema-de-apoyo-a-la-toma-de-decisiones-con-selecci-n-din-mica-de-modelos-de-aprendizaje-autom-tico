import gradio as gr
import pandas as pd
import cutie
import os
import auxiliary_fun
import static_frame as sf
from termcolor import colored
import numpy as np
import cutie


import DprepNcleaning
import eda
import Mbuilding
import dtale

##########################    ##########################    ##########################    ##########################    ##########################    ##########################
##########################    ##########################    ##########################    ##########################    ##########################    ##########################


def var_acquisition(column_name):

    global DATA
    global CHECK #sacar de aca
    global TARGET_TYPE
    global drop_target_col_values #needed for the scewed cast
    global TARGET_COLUMN #needed for data drop in scewed data
    TARGET_TYPE = 'Not identify'
    # LEN_SCEWED_2 = False
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

    return f"You have selected the column: {TARGET_TYPE}"

def check_selection(selection):

    global CHECK
    CHECK = selection

    return CHECK

def scewed_selection(selection):

    global DATA
    global TARGET_TYPE
    global TARGET_COLUMN
    global drop_target_col_values
    if selection == "Yes":
        print('dropeooooo si siiiii funcion')
        TARGET_TYPE = 'boolean'
        for drop_val in drop_target_col_values:
            drop_row = DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==drop_val].index
            DATA = DATA.drop(drop_row)

    return CHECK



# Define a function for the first tab
def greet(name):
    return f"Hello, {name}!"

# Define a function for the second tab
def calculate_square(number):
    return number ** 2




##########################    ##########################    ##########################    ##########################    ##########################    ##########################
##########################    ##########################    ##########################    ##########################    ##########################    ##########################




# Create a Gradio interface
with gr.Blocks() as demo:
    with gr.Tabs():
        print(colored('outside','red'))
        DATAcasted = 'Not uploaded'
        with gr.Tab(label="1 tab"):
            print(colored('1 tab','red'))
            print('\nInitialize -> gr.BLOCK \n')
            # define FLAGS for render order
            column_dropdown_RFLAG = gr.State(0)
            target_check_RFLAG = gr.State(0) #need to be global?
            show_type_RFLAG = gr.State(0)

            gr.Markdown("## Upload dataset and select target column")
            data = gr.File(label="Upload CSV / XLSX file", type="filepath") #data id the file uploaded

            @gr.render(inputs=[data])
            def show_split(FILE):

                print('1 - Entered to render of auto detection check <DATA>')
                #enter if a file is loaded (define DATA and its columns Variables_OF_DATA)
                if FILE != None: 

                    print('----FILE != None')
                    global DATA
                    global VARIABLES_OF_DATA
                    global target_check_RFLAG 
                    global target_check_btn
                    if FILE.name[len(FILE.name)-5:] == '.xlsx':
                        DATA = pd.read_excel(FILE.name)
                    if FILE.name[len(FILE.name)-4:] == '.csv':
                        DATA = pd.read_csv(FILE.name)
                    VARIABLES_OF_DATA = DATA.columns #used in target column acq
                    gr.Textbox(DATA.head(), label="The head of the dataset is:")

                    target_check_btn = gr.Radio(["Yes", "No"], label="User check of auto detection of target column:")
                    #change CHECK when press
                    target_check_btn.input(fn=check_selection, inputs=target_check_btn, outputs=None)
                    #change target_check_RFLAG <for dissable of the button funcionality>
                    target_check_btn.input(lambda count: count + 1, target_check_RFLAG, target_check_RFLAG, scroll_to_output=True)



            @gr.render(inputs=[target_check_RFLAG])
            def show_split(TARGET_CHECK):

                print('2 - Entered to render of target selection <TARGET CHECK>')
                #enters when we select if we want the user verification of the target type
                if TARGET_CHECK>0:
                    print('----TARGET_CHECK>0')
                    print(DATA) 
                    gr.Markdown("Select a variable to predict ")
                    column_dropdown = gr.Dropdown(list(VARIABLES_OF_DATA), label="Please select the varaible to predict from the next list: ", filterable=False)
                    #filterable container
                    # Set the function to be called when the dropdown value changes
                    column_dropdown.change(fn=var_acquisition, inputs=column_dropdown, outputs=None, scroll_to_output=True)
                    column_dropdown.input(lambda count: count + 1, column_dropdown_RFLAG, column_dropdown_RFLAG, scroll_to_output=True)
                    
            @gr.render(inputs=[column_dropdown_RFLAG])
            def scewed_data_cast(column_dropdown):

                SCEWED_FLAG = False
                print('4 - Enter to render of type detection <CLUMN DROPDOWN>')
                if column_dropdown>1:
                    print(colored('---column_dropdown activate', 'red'))
                    print(colored(DATA, 'red'))
                    

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
                        print(colored('scewed data', 'blue'))
                    #if we dont hace a scewed data -> WE SHOW THE RESULT <SAME AS LINE 173>
                    else: 
                        print(colored('else scewed data', 'blue'))
                        # print(TARGET_TYPE)
                        print(colored(DATA, 'yellow'))
                        gr.Textbox(TARGET_TYPE, label="The target type is:")
                        # print(CHECK)

                        #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
                        # global DATAcasted
                        DATA_casted = auxiliary_fun.d_cast(DATA, TARGET_COLUMN, TARGET_TYPE)
                        print(colored(DATA_casted, 'yellow'))

                            ### Step 3.1: Exploratory Data Analyzis (MANUAL)###
                        manualEDA, missing4rows = eda.ManualEDAfun(DATA_casted) #data no tiene a predicted column
                        # print(colored('\nTable with information of the variables:', 'red', attrs=['bold']))
                        # print(colored(manualEDA, 'red'))
                        # print(colored('\nTable with information of the rows:', 'red', attrs=['bold']))
                        # print(colored(missing4rows, 'red'))
                        
                            ### Step 2: Data Cleaning ###    
                        #min_porcentage_col if missing>10 for a column, we drop it
                        DATA_cleaned = DprepNcleaning.data_cleaning(DATA_casted, min_porcentage_col = 10, min_porcentage_row = 0)
                        # print(colored('\nThe result of the number of patients is: '+str(len(data)), 'red', attrs=['bold']))
                        print(colored(DATA_cleaned, 'red'))

                    # print(CHECK) implementar despues

                    if SCEWED_FLAG == True:
                        print(colored('drop scewed question', 'blue'))
                        scewed_yes_no_btn = gr.Radio(["Yes", "No"], label="Cast to boolean (once selected there is no undo):")
                        scewed_yes_no_btn.input(fn=scewed_selection, inputs=scewed_yes_no_btn, outputs=None) #here we cast DATA (when the button is press there is not way back)
                        scewed_yes_no_btn.input(lambda count: count + 1, show_type_RFLAG, show_type_RFLAG, scroll_to_output=True)

            # to show the result in the case of scewed data (cast or not cast) <SAME AS LINE 160>
            @gr.render(inputs=[show_type_RFLAG])
            def scewed_data_cast(SHOW_TYPE):
                if SHOW_TYPE>0:
                    print('user maked a selection')
                    # print(TARGET_TYPE)
                    print(colored(DATA, 'blue'))
                    gr.Textbox(TARGET_TYPE, label="The target type is:")
                    # print(CHECK) implementar despues
                    
                    #AUTO-CAST strings to int (if <unique == 1> we save the value as a key and drop the column for the model)
                    # global DATAcasted
                    DATAcasted = auxiliary_fun.d_cast(DATA, TARGET_COLUMN, TARGET_TYPE)
                    print(colored(DATAcasted, 'yellow'))

        
        with gr.Tab(label="Square Calculator Tab"):
            print(colored('2 tab','red'))

            EDA_RFLAG = gr.State(0)
            number_input = gr.Number(label="Enter a number")
            calculate_button = gr.Button("Calculate Square")
            square_output = gr.Label()
            calculate_button.click(lambda count: count + 1, EDA_RFLAG, EDA_RFLAG, scroll_to_output=True)
            # gr.Label(DATAcasted)

            @gr.render(inputs=[EDA_RFLAG])
            def scewed_data_cast(EDA):
                if EDA>0:
                    gr.Label(DATAcasted)

# Launch the Gradio interface
demo.launch()
