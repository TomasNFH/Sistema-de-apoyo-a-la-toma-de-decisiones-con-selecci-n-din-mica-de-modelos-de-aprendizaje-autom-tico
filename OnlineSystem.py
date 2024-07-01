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
def d_acquisition(FILE):

    global DATA
    global VARIABLES_OF_DATA
    if FILE.name[len(FILE.name)-5:] == '.xlsx':
        DATA = pd.read_excel(FILE.name)
    if FILE.name[len(FILE.name)-4:] == '.csv':
        DATA = pd.read_csv(FILE.name)
    VARIABLES_OF_DATA = DATA.columns #used in target column acq
    
    return DATA.head()

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
            print('changed state od len scewed 2')

        # print(LEN_SCEWED_2)

    return f"You have selected the column: {TARGET_TYPE}"

def check_selection(selection):
    global CHECK
    CHECK = selection
    print(CHECK)
    return CHECK


def scewed_selection(selection):
    # global CHECK
    # CHECK = selection
    global DATA
    global TARGET_TYPE
    global TARGET_COLUMN
    global drop_target_col_values

    print('\n\nscewed sssss')
    print(selection)
    if selection == "Yes":
        print('CASTEOOOOO NAZZZZIII')
        TARGET_TYPE = 'boolean'
        for drop_val in drop_target_col_values:
            drop_row = DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==drop_val].index
            DATA = DATA.drop(drop_row)



    else:
        print('notttt')
    return CHECK

def disable_buttons(selection):
    scewed_yes_no_btn.update(interactive=False)
    return check_selection(selection)

# FLAG_UPLOAD_FILE = False
with gr.Blocks() as upload_fileTAB:
    target_flag = gr.State(0) #if is 0, we dont enter in target selection in the TAB
    len_sc_2 = gr.State(0)

    gr.Markdown("## Upload dataset and select target column")
    data = gr.File(label="Upload CSV / XLSX file", type="filepath") #data id the file uploaded
    # Set the function to be called when the data value changes
    data.change(d_acquisition, data, gr.Textbox(label="The head of the dataset is:"))

    target_btn = gr.Button("Select target (after upload of dataset)")
    target_btn.click(lambda count: count + 1, target_flag, target_flag)
    # @gr.render(inputs=data)
    @gr.render(inputs=[target_flag])
    def show_split(TARGET_FLAG):
        print('entered to render')
        if TARGET_FLAG==0:
            gr.Markdown(" ")
        if TARGET_FLAG>0:
            target_check_btn = gr.Radio(["Yes", "No"], label="User check of auto detection of target column:", value="Yes")
            target_check_btn.change(fn=check_selection, inputs=target_check_btn, outputs=None)
            # print(CHECK)

            gr.Markdown("Select a variable to predict ")
            # gr.Dropdown(list(var_acquisition), label="Please select the varaible to predict from the next list: ")
            column_dropdown = gr.Dropdown(list(VARIABLES_OF_DATA), label="Please select the varaible to predict from the next list: ")


            # Set the function to be called when the dropdown value changes
            column_dropdown.change(fn=var_acquisition, inputs=column_dropdown, outputs=None)

            check_type_btn = gr.Button("Check the target type (after upload of dataset)")
            check_type_btn.click(lambda count: count + 1, len_sc_2, len_sc_2)

    @gr.render(inputs=[len_sc_2, target_flag])
    def scewed_data_cast(LEN_SCEWED_2_RENDER, TARGET_FLAG):
        print('entr render 2')
        # global TARGET_FLAG

        # if TARGET_FLAG>1: #hacer q aparesca despues de scewed menu 
            # gr.Label(f"The target type is: {TARGET_TYPE}")

        if LEN_SCEWED_2_RENDER==1:
            print('cast??')
            scewed_yes_no_btn = gr.Radio(["Yes", "No"], label="Cast to boolean (once selected there is no undo):")
            scewed_yes_no_btn.change(fn=scewed_selection, inputs=scewed_yes_no_btn, outputs=None) #here we cast DATA (when the button is press there is not way back)
            # scewed_yes_no_btn.change(fn=disable_buttons, inputs=scewed_yes_no_btn, outputs=gr.Label())
            


######### Support decision system #########
bye_world = gr.Interface(lambda name: "Bye " + name, "text", "text")



######### ALL #########
demo = gr.TabbedInterface([upload_fileTAB, bye_world], ["Dynamic seleccion of machine learning models", "Support decision system"])

if __name__ == "__main__":
    demo.launch()