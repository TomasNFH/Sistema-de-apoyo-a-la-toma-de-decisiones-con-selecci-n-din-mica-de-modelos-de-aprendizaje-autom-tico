import gradio as gr
import pandas as pd
import cutie
import os
import auxiliary_fun
import static_frame as sf
from termcolor import colored
import numpy as np
import cutie
# import aux_fun_online


##########################    ##########################    ##########################    ##########################    ##########################    ##########################
##########################    ##########################    ##########################    ##########################    ##########################    ##########################


def var_acquisition(column_name):

    global DATA
    global CHECK #sacar de aca
    global TARGET_TYPE
    global len_scewed_RFLAG #ver if we acumulated in two values only, if true we ask in tab what to do (in this function we define the class a classes and don do anythong)
    global drop_target_col_values #needed for the scewed cast
    global TARGET_COLUMN #needed for data drop in scewed data
    TARGET_TYPE = 'Not identify'
    # LEN_SCEWED_2 = False
    TARGET_COLUMN = column_name
    #check if the user already selected something
    if len(TARGET_COLUMN) != 0: #check if the
        print(colored('inside of var acq fun', 'red'))
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
            len_scewed_RFLAG = gr.State(1)
            print(colored('scewed data', 'blue'))

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
        print('dropeooooo si siiiii')
        TARGET_TYPE = 'boolean'
        for drop_val in drop_target_col_values:
            drop_row = DATA[TARGET_COLUMN][DATA[TARGET_COLUMN]==drop_val].index
            DATA = DATA.drop(drop_row)

    return CHECK

# def check_state(state):
#     if state != 0:
#         message = f"State was not zero, button clicked {state} times"
#         print(message)
#         ret_FLAG = 1
#     else:
#         message = "State is zero, no action taken"
#         print(message)
#         ret_FLAG = 0
#     return ret_FLAG
##########################    ##########################    ##########################    ##########################    ##########################    ##########################
##########################    ##########################    ##########################    ##########################    ##########################    ##########################


with gr.Blocks() as upload_fileTAB:

    print('\nInitialize -> gr.BLOCK \n')
    # define FLAGS for render order
    # global target_check_RFLAG
    global len_scewed_RFLAG
    column_dropdown_RFLAG = gr.State(0)
    target_check_RFLAG = gr.State(0) #need to be global?
    len_scewed_RFLAG = gr.State(0)

    print()
    gr.Markdown("## Upload dataset and select target column")
    data = gr.File(label="Upload CSV / XLSX file", type="filepath") #data id the file uploaded
    #ENTERS ONLY IF data CHANGES
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
        # global target_check_btn
        # global VARIABLES_OF_DATA
        #enters when we select if we want the user verification of the target type
        if TARGET_CHECK>0:
            print('----TARGET_CHECK>0')
            gr.Markdown("Select a variable to predict ")
            column_dropdown = gr.Dropdown(list(VARIABLES_OF_DATA), label="Please select the varaible to predict from the next list: ", filterable=False)
            #filterable container
            # Set the function to be called when the dropdown value changes
            column_dropdown.change(fn=var_acquisition, inputs=column_dropdown, outputs=None, scroll_to_output=True)
            column_dropdown.input(lambda count: count + 1, column_dropdown_RFLAG, column_dropdown_RFLAG, scroll_to_output=True)
            
    @gr.render(inputs=[column_dropdown_RFLAG])
    def scewed_data_cast(column_dropdown):
        # auxFLAG = False
        # PRINT_RET_FLAG = False
        SCEWED_FLAG = False
        print('4 - Enter to render of type detection <CLUMN DROPDOWN>')
        if column_dropdown>1:
            
            print(colored('column_dropdown', 'red'))
            ######
            
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
                # len_scewed_RFLAG = gr.State(1)
                SCEWED_FLAG= True
                print(colored('scewed data', 'blue'))
            else: 
                print(colored('else', 'blue'))
                # PRINT_RET_FLAG = True

            ######

            if SCEWED_FLAG == True:
                # print('ahora')

                scewed_yes_no_btn = gr.Radio(["Yes", "No"], label="Cast to boolean (once selected there is no undo):")
                scewed_yes_no_btn.change(fn=scewed_selection, inputs=scewed_yes_no_btn, outputs=None) #here we cast DATA (when the button is press there is not way back)
                # PRINT_RET_FLAG = True


            # if PRINT_RET_FLAG==True:
            #     gr.Markdown(TARGET_TYPE)

            # scewed_yes_no_btn.change(fn=disable_buttons, inputs=scewed_yes_no_btn, outputs=gr.Label())

    # # we add column_dropdown_RFLAG to activate the render becouse we change the value of len_scewed_RFLAG in a function (dowsnt trigger render)
    # @gr.render(inputs=[len_scewed_RFLAG, column_dropdown_RFLAG])
    # def scewed_data_cast(LEN_SCEWED, NOT_USE):

    #     print('3 - Enter to render scewed selection <LEN SCEWED>')
    #     # aux = LEN_SCEWED.change(fn=LEN_SCEWED, input = LEN_SCEWED, output=None)
    #     print(colored(LEN_SCEWED, 'green'))

    #     if LEN_SCEWED==0:
    #         print(colored('not changed yet','blue'))
    #     if LEN_SCEWED>0: #add other condition
    #         scewed_yes_no_btn = gr.Radio(["Yes", "No"], label="Cast to boolean (once selected there is no undo):")
    #         scewed_yes_no_btn.change(fn=scewed_selection, inputs=scewed_yes_no_btn, outputs=None) #here we cast DATA (when the button is press there is not way back)
    #         # scewed_yes_no_btn.change(fn=disable_buttons, inputs=scewed_yes_no_btn, outputs=gr.Label())




            # print(colored(CHECK, 'red'))
            # print(colored(TARGET_TYPE, 'red'))
            # print(colored(TARGET_TYPE, 'red'))

        # if len_scewed_RFLAG>0: #add other condition
            # print(colored('enters in tpe detection render','blue'))



######### Support decision system #########
bye_world = gr.Interface(lambda name: "Bye " + name, "text", "text")



######### ALL #########
demo = gr.TabbedInterface([upload_fileTAB, bye_world], ["Data acquisition", "EDA"])

# resultados del modelo
#pestañaa para preddecir


if __name__ == "__main__":
    demo.launch()