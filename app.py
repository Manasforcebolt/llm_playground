from langchain_community.chat_models import ChatOllama
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
import time
import streamlit as st

st.set_page_config(layout="wide")

st.title("Open Source LLMs Playground")
user_input = None
template = None

with st.sidebar :
    st.header(":red[Model]  Configurations")
    model = st.selectbox("Choose your model",
                         ("llama2","llama2:7b-chat","mistral", "notus","llava","codellama","starling-lm","More are coming soon ....")
                        )
    temperature = st.slider("Temparature" , 0.0, 1.0, 0.2)
    repeat_penalty = st.slider("Penalty for repeated words", 1.0, 2.0, 1.1)
    num_predict = st.slider("Maximum number of tokens in generating",100, 1048, 256)

    # st.markdown("\n\n\n\nSYSTEM")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    # global template 
    template = st.text_area(label="Prompt",placeholder="You are an helpful assistant.") 



if model =="llama2:7b-chat":
    llm = ChatOllama(model=model,
                     temperature = temperature,
                     repeat_penalty = repeat_penalty,
                     num_predict = num_predict)

llm = Ollama(model=model,
            temperature = temperature,
            repeat_penalty = repeat_penalty,
            # num_predict = num_predict
            )




def Chat(query):
    # global template
    prompt = PromptTemplate(input_variables = ["input"],
                            template = template + "{input}")
    llm_chain = LLMChain(
    llm=llm,
    prompt = prompt,
    # memory=ConversationBufferMemory(30)
    )

    res = llm_chain.run(input=query, template = prompt)
    return res

    

def stream_data(res):
    for w in res.split():
        yield w + " "
        time.sleep(0.09)    





if 'user_input' not in st.session_state:
    st.session_state['user_input']=[]

if 'ai_response' not in st.session_state:
    st.session_state['ai_response']=[]

def get_text():
    input_text = st.text_input("User Message", key="input")
    return input_text

# global user_input 
user_input = get_text()


if user_input:
        output = st.write_stream(stream_data(Chat(user_input)))
        st.session_state.user_input.append(user_input)
        st.session_state.ai_response.append(output)

message_history = st.empty()
 
if st.session_state['user_input']:
    for i in range(0,len(st.session_state['user_input']),1):
        
        message(st.session_state["user_input"][i], 
                key=str(i),avatar_style="miniavs",is_user=True)
        
        message(st.session_state['ai_response'][i], 
                avatar_style="icons",
                key=str(i) + 'data_by_user')

    
