import gradio as gr
import pandas as pd
import cutie

import os
import auxiliary_fun
import static_frame as sf
from termcolor import colored
import numpy as np
import cutie

######### Dynamic seleccion of machine learning models #########
def var_acquisition(column_name):

    global DATA
    global CHECK #sacar de aca
    global TARGET_TYPE
    global LEN_SCEWED_2 #ver if we acumulated in two values only, if true we ask in tab what to do (in this function we define the class a classes and don do anythong)
    global drop_target_col_values #needed for the scewed cast
    global TARGET_COLUMN
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
        
        ### 3.1 verify scewed predict data (to cast to boolean)
        SCEWED_TARGET_COL = pd.DataFrame(columns = ['Value id', 'Value', 'Count', 'Occurences (%)'])
        len_target_col = len(DATA[TARGET_COLUMN])
        for k,unique_val in enumerate(unique_values):
            unique_count = len(DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==unique_val]) 
            SCEWED_TARGET_COL.loc[len(SCEWED_TARGET_COL.index)] = [k, unique_val, unique_count, (unique_count/len_target_col)*100]
        drop_target_col_values = SCEWED_TARGET_COL[SCEWED_TARGET_COL['Occurences (%)']<10]['Value'] 
        target_col_values = SCEWED_TARGET_COL[SCEWED_TARGET_COL['Occurences (%)']>=10]['Value'] 
        
        ###In the case that values concentrate in only two values and we have something to drop <unque!=2>, ask if the user want to drop the columns that dont have much samples (making it a boolean) 
        if len(target_col_values) == 2 and len(unique_values)!=2:
            LEN_SCEWED_2 = gr.State(1)

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
        TARGET_TYPE = 'boolean'
        for drop_val in drop_target_col_values:
            drop_row = DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==drop_val].index
            DATA = DATA.drop(drop_row)

    return CHECK






# FLAG_UPLOAD_FILE = False
with gr.Blocks() as upload_fileTAB:
    # global target_flag
    global check_disable_RFLAG

    print('0 - gr BLOCK entry')
    # target_flag = gr.State(0) #if is 0, we dont enter in target selection in the TAB
    # len_sc_2 = gr.State(0)
    column_dropdown_RFLAG = gr.State(0)
    #A FLAG that is used to disable CHECK gr.Radio
    check_disable_RFLAG = gr.State(0)

    gr.Markdown("## Upload dataset and select target column")
    data = gr.File(label="Upload CSV / XLSX file", type="filepath") #data id the file uploaded

    #ENTERS ONLY IF data CHANGES
    @gr.render(inputs=[data])
    def show_split(FILE):
        global DATA
        global VARIABLES_OF_DATA
        global check_disable_RFLAG
        global target_check_btn
        print('1 - Entered to render of auto detection check <DATA>')

        #enter if a file is loaded (define DATA and its columns Variables_OF_DATA)
        if FILE != None: 
            print('FILE != None')
            global DATA
            global VARIABLES_OF_DATA
            if FILE.name[len(FILE.name)-5:] == '.xlsx':
                DATA = pd.read_excel(FILE.name)
            if FILE.name[len(FILE.name)-4:] == '.csv':
                DATA = pd.read_csv(FILE.name)
            VARIABLES_OF_DATA = DATA.columns #used in target column acq
            gr.Textbox(DATA.head(), label="The head of the dataset is:")

            target_check_btn = gr.Radio(["Yes", "No"], label="User check of auto detection of target column:")
            #change CHECK when press
            target_check_btn.input(fn=check_selection, inputs=target_check_btn, outputs=None)
            #change check_disable_RFLAG <for dissable of the button funcionality>
            target_check_btn.input(lambda count: count + 1, check_disable_RFLAG, check_disable_RFLAG, scroll_to_output=True)
            # print('print(target_check_btn)')
            # print(target_check_btn)

    @gr.render(inputs=[check_disable_RFLAG])
    def show_split(CHECK_DISABLE):

        global target_check_btn
        global VARIABLES_OF_DATA
        print('2 - Entered to render of target selection <DISABLE FLAG>')

        #enters when we select if we want the user verification of the target type
        if CHECK_DISABLE>0:
            print('CHECK_DISABLE>0')
            # target_check_btn = gr.Radio(["Yes", "No"], label="User check of auto detection of target column:", value="Yes")
            gr.Markdown("Select a variable to predict ")
            column_dropdown = gr.Dropdown(list(VARIABLES_OF_DATA), label="Please select the varaible to predict from the next list: ")
            #filterable container
            # Set the function to be called when the dropdown value changes
            column_dropdown.change(fn=var_acquisition, inputs=column_dropdown, outputs=None, scroll_to_output=True)
            column_dropdown.input(lambda count: count + 1, column_dropdown_RFLAG, column_dropdown_RFLAG, scroll_to_output=True)

            # check_type_btn = gr.Button("Check the target type (after upload of dataset)")
            # check_type_btn.click(lambda count: count + 1, len_sc_2, len_sc_2, scroll_to_output=True)

    @gr.render(inputs=[column_dropdown_RFLAG])
    def scewed_data_cast(LEN_SCEWED_2_RENDER):
        print('3 - Enter to render <SCEWED>')

        # if TARGET_FLAG>1: #hacer q aparesca despues de scewed menu 
            # gr.Label(f"The target type is: {TARGET_TYPE}")

        if LEN_SCEWED_2_RENDER>1: #add other condition
            scewed_yes_no_btn = gr.Radio(["Yes", "No"], label="Cast to boolean (once selected there is no undo):")
            scewed_yes_no_btn.change(fn=scewed_selection, inputs=scewed_yes_no_btn, outputs=None) #here we cast DATA (when the button is press there is not way back)
            # scewed_yes_no_btn.change(fn=disable_buttons, inputs=scewed_yes_no_btn, outputs=gr.Label())
            


######### Support decision system #########
bye_world = gr.Interface(lambda name: "Bye " + name, "text", "text")



######### ALL #########
demo = gr.TabbedInterface([upload_fileTAB, bye_world], ["Data acquisition", "EDA"])

# resultados del modelo
#pestañaa para preddecir


if __name__ == "__main__":
    demo.launch()