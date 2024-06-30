import gradio as gr
import pandas as pd


######### Dynamic seleccion of machine learning models #########
def file_to_df(FILE):

    global DATA
    global var_acquisition 
    if FILE.name[len(FILE.name)-5:] == '.xlsx':
        DATA = pd.read_excel(FILE.name)
    if FILE.name[len(FILE.name)-4:] == '.csv':
        DATA = pd.read_csv(FILE.name)
    var_acquisition = DATA.columns #used in target column acq
    
    return DATA.head()

def file_columns(FILE):
    if FILE.name[len(FILE.name)-5:] == '.xlsx':
        DATA = pd.read_excel(FILE.name)
    if FILE.name[len(FILE.name)-4:] == '.csv':
        DATA = pd.read_csv(FILE.name)
    return str(DATA.columns)

def select_column(column_name):
    aux = column_name
    print('AUXXX')
    print(aux)

    return f"You have selected the column: {column_name}"

# FLAG_UPLOAD_FILE = False
with gr.Blocks() as upload_fileTAB:
    track_count = gr.State(0)

    gr.Markdown("## Upload dataset and select target column")
    data = gr.File(label="Upload CSV / XLSX file", type="filepath") #data id the file uploaded
    # Set the function to be called when the data value changes
    data.change(file_to_df, data, gr.Textbox(label="Resultado de carga"))
    target_btn = gr.Button("Select target")
    target_btn.click(lambda count: count + 1, track_count, track_count)

    # @gr.render(inputs=data)
    @gr.render(inputs=[track_count])
    def show_split(track_count):
        if track_count==0:
            gr.Markdown(" ")
        else:
            gr.Markdown("Select a variable to predict ")
            print(list(var_acquisition))
            # gr.Dropdown(list(var_acquisition), label="Please select the varaible to predict from the next list: ")
            column_dropdown = gr.Dropdown(list(var_acquisition), label="Please select the varaible to predict from the next list: ")
            print(column_dropdown)

            output = gr.Label()
            # Set the function to be called when the dropdown value changes
            column_dropdown.change(fn=select_column, inputs=column_dropdown, outputs=output)
######### Tab 2


# hello_world = gr.Interface(lambda name: "Hello " + name, "text", "text")
bye_world = gr.Interface(lambda name: "Bye " + name, "text", "text")



##
demo = gr.TabbedInterface([upload_fileTAB, bye_world], ["Dynamic seleccion of machine learning models", "Support decision system"])


if __name__ == "__main__":
    demo.launch()