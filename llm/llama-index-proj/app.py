import gradio as gr
from rag_system import RAGSystem
import os
from config import DOCS_DIR

rag = RAGSystem()

def answer_question(question: str) -> str:
    try:
        return rag.query(question)
    except Exception as e:
        return f"发生错误: {str(e)}"

def upload_file(file):
    try:
        file_path = os.path.join(DOCS_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        rag.add_document(file_path)
        return f"文件 {file.name} 已成功添加到知识库"
    except Exception as e:
        return f"上传失败: {str(e)}"

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 本地知识库问答系统")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="上传文档到知识库")
            upload_button = gr.Button("上传")
            upload_output = gr.Textbox(label="上传状态")
        
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="请输入您的问题")
            submit_btn = gr.Button("提交问题")
            answer_output = gr.Textbox(label="回答")

    upload_button.click(
        fn=upload_file,
        inputs=[file_input],
        outputs=[upload_output]
    )
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
