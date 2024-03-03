# TODO:
# - accelerated inference
# - collab migration
# - function calling
# - whisper integration

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain_community.llms import (
    CTransformers,
    LlamaCpp
)
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler


from pydantic import BaseModel
from json import dump

import chainlit as cl

class LlmConfig(BaseModel):
    ctx_size: int
    out_size: int
    temperature: float
    top_k: int
    top_p: float
    repeat_penalty: float
    seed: int
    threads: int
    gpu_layers: int
    batch_size: int
    model: str
    m_type: str
    template: str
    system_message: str

class Llm: 
  def get(self):
      return self.llm
        
class LlmLlamaCpp(Llm):
    def __init__(self, conf: LlmConfig):
        self.conf = conf
        self.llm = LlamaCpp(
            model_path = conf.model,
            n_ctx = conf.ctx_size,
            max_tokens = conf.out_size,
            temperature =conf.temperature, 
            top_k = conf.top_k,
            top_p = conf.top_p,
            repeat_penalty = conf.repeat_penalty,
            seed = conf.seed,
            n_threads = conf.threads,
            n_gpu_layers = conf.gpu_layers,
            n_batch = conf.batch_size,
            callbacks = [StreamingStdOutCallbackHandler()],
            streaming = True,
            verbose = True
        )
class LlmCtransformers(Llm):
    def __init__(self, conf: LlmConfig):
        self.conf = conf
        self.llm = CTransformers(
            model = conf.model,
            model_type = conf.m_type,
            config = {
                "context_length": conf.ctx_size,
                "max_new_tokens": conf.out_size,
                "temperature":conf.temperature, 
                "top_k": conf.top_k,
                "top_p": conf.top_p,
                "repetition_penalty": conf.repeat_penalty,
                "seed": conf.seed,
                "threads": conf.threads,
                "gpu_layers": conf.gpu_layers
            },
            callbacks = [StreamingStdOutCallbackHandler()],
            streaming = True
        )

def create_chain(llm: Llm, conf: LlmConfig):
    prompt = ChatPromptTemplate.from_template(conf.template)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        return_messages=False)
    chain =  LLMChain(
        prompt=prompt,
        llm=llm.get(),
        memory=memory,
        verbose=True)
    return chain

codellama_template = """
<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{chat_history}

{user_message} [/INST]
"""

llama_conf = LlmConfig(
    ctx_size = 2048,
    out_size = 2048,
    temperature = 0.9,
    top_k = 40,
    top_p = 0.75,
    repeat_penalty = 1.12,
    seed = 112348,
    threads = 4,
    gpu_layers = 28,
    batch_size = 512,
    #model = "../../hf/bloke/codellama-13b-instruct.Q4_K_M.gguf",
    model = "../../hf/bloke/codellama-7b-instruct.Q5_K_M.gguf",
    #model = "../../hf/bloke/codellama-7b-instruct.Q4_K_M.gguf",
    m_type = "llama",
    template = codellama_template,
    system_message = "You are an expert programmer who can write complete and correct programs in java, scala and rust, using best practices and latest standards. Generated code must be readable and well formatted using markdown."
)

def run():
    llama = LlmLlamaCpp(llama_conf)
    chain = create_chain(llama, llama_conf)    
    chain.invoke({
        "system_message": llama_conf.system_message,
        "user_message": "write a scala program to print primes upto 1M."
    })
    
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")

def store_qna(msg, file="chat_history.json"):
    with open(file, 'a') as f:
        dump(msg, f)
        f.write("\n")

@cl.on_chat_start
async def on_chat_start():
    llama = LlmLlamaCpp(llama_conf)
    chain = create_chain(llama, llama_conf)    
    cl.user_session.set("runnable", chain)
    cl.user_session.set("conf", llama_conf)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"system_message": llama_conf.system_message, "user_message": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(), StreamHandler()]),
    ):
        #await msg.stream_token(chunk)
        chunk["conf"] = llama_conf.model_dump()
        store_qna(chunk)
    await msg.send()

    