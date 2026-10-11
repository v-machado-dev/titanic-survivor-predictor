import gradio as gr
import joblib as jb
import spaces

@spaces.GPU
def predict(sex, age, pclass):
    model = jb.load('ml_web.pkl')
    pclass = int(pclass)
    p = model.predict_proba([[sex, age, pclass]])[0]

    return {'Did not survive': p[0], 'Survived': p[1]}

demo = gr.Interface(fn=predict,
                    inputs=[
                        gr.Dropdown(choices=['Male', 'Female'], type='index',
                                    label='Genre', info="Passenger's Genre"),            
                        gr.Number(label='Age', info="Passenger's Age"),
                        gr.Dropdown(choices=['1', '2', '3'], type='value',
                                    label='Class', info="Ticket class: 1 = First Class, 2 = Second Class, 3 = Third Class"),
                                    ],
                        outputs=gr.Label(label='Forecasting'),
                        title= "Titanic Passenger Survival Predictor")

demo.launch(ssr_mode=False)
