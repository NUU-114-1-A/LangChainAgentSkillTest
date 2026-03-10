import os
import pandas as pd
import requests
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ==========================================
# 0. 設定目錄結構的絕對路徑
# ==========================================
KNOWLEDGE_BASE_DIR = r"C:\rag-skill\rag-skill\knowledge"

# ==========================================
# 1. 定義適配該結構的真實工具 (Tools)
# ==========================================
@tool
def list_directory(relative_path: str = "") -> str:
    """
    探索目錄結構。
    傳入相對路徑 (例如 "" 探索根目錄，"AI Knowledge" 探索 AI 報告目錄)。
    回傳該目錄下的所有檔案與資料夾清單。
    """
    target_path = os.path.join(KNOWLEDGE_BASE_DIR, relative_path)
    try:
        items = os.listdir(target_path)
        return f"📁 目錄 '{relative_path}' 下的內容：\n" + "\n".join(items)
    except FileNotFoundError:
        return f"❌ 錯誤：找不到目錄 '{relative_path}'"
    except Exception as e:
        return f"❌ 讀取目錄發生錯誤：{str(e)}"
@tool
def trigger_n8n_workflow(action: str, message: str) -> str:
    """
    【系統強制要求】當你需要「控制實體設備(如開燈/風扇)」或「發送緊急通知/警報」給家屬時，必須呼叫此工具。
    action: 欲執行的動作指令 (例如 'turn_on_light', 'send_alert')
    message: 要傳遞給 n8n 的詳細訊息或理由
    """
    
    webhook_url = "http://localhost:5678/webhook-test/96f8b909-0c4c-4f6b-9915-33236dc42d9f" 
    
    payload = {
        "action": action,
        "message": message
    }
    
    try:
        print(f"\n[系統底層] 正在呼叫 n8n 工作流... 動作: {action}")
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 200:
            return f"✅ 成功觸發 n8n！已經執行：{action}。"
        else:
            return f"❌ 呼叫 n8n 失敗，狀態碼：{response.status_code}"
    except Exception as e:
        return f"❌ 連線 n8n 發生錯誤：{str(e)}"
@tool
def read_text_file(relative_file_path: str, num_lines: int = 100) -> str:
    """
    讀取 Markdown (.md) 或純文字檔的內容。
    【特別重要】：到達任何新目錄時，必須優先用此工具讀取該目錄下的 `data_structure.md`！
    傳入相對路徑 (例如 "data_structure.md" 或 "Safety Knowledge/data_structure.md")。
    """
    target_path = os.path.join(KNOWLEDGE_BASE_DIR, relative_file_path)
    
    
    encodings = ['utf-8', 'big5', 'gbk', 'cp950']
    
    content = None
    for enc in encodings:
        try:
            with open(target_path, 'r', encoding=enc) as f:
                lines = f.readlines()
                content = lines
                break 
        except (UnicodeDecodeError, UnicodeError):
            continue 
            
    if content is None:
        return f"❌ 錯誤：無法以任何已知編碼讀取檔案 '{relative_file_path}'。"

    # 判斷檔案行數
    if len(content) <= num_lines:
        return f"📄 {relative_file_path} 的完整內容：\n" + "".join(content)
    else:
        return f"📄 {relative_file_path} 的前 {num_lines} 行內容：\n" + "".join(content[:num_lines])


@tool
def read_pdf_preview(relative_file_path: str, page_num: int = 0) -> str:
    """
    讀取 PDF 檔案的特定頁面內容 (預設讀取第 0 頁，也就是第一頁)。
    傳入相對路徑 (例如 "Knowledge/目標路徑.pdf")。
    可用來掃描 PDF 的目錄或摘要。
    """
    target_path = os.path.join(KNOWLEDGE_BASE_DIR, relative_file_path)
    try:
        reader = PdfReader(target_path)
        if page_num >= len(reader.pages):
            return f"❌ 錯誤：該 PDF 只有 {len(reader.pages)} 頁。"
        
        page = reader.pages[page_num]
        text = page.extract_text()
        return f"📑 {relative_file_path} (第 {page_num} 頁) 內容擷取：\n{text[:1500]}..." 
    except Exception as e:
        return f"❌ 讀取 PDF 發生錯誤：{str(e)}"

@tool
def read_excel_preview(relative_file_path: str, num_rows: int = 5) -> str:
    """
    分析 Excel (.xlsx) 結構化資料。
    傳入相對路徑 (例如 "E-commerce Data/inventory.xlsx")。
    回傳 Excel 的欄位結構與前幾筆資料預覽。
    """
    target_path = os.path.join(KNOWLEDGE_BASE_DIR, relative_file_path)
    try:
        df = pd.read_excel(target_path)
        info = f"📊 Excel 檔案: {relative_file_path}\n"
        info += f"總資料筆數: {len(df)}\n"
        info += f"包含欄位: {', '.join(df.columns)}\n"
        info += f"前 {num_rows} 筆預覽:\n{df.head(num_rows).to_markdown()}"
        return info
    except Exception as e:
        return f"❌ 讀取 Excel 發生錯誤：{str(e)}"

tools = [list_directory, read_text_file, read_pdf_preview, read_excel_preview,trigger_n8n_workflow]

# ==========================================
# 2. 連接 LM Studio 
# ==========================================
llm = ChatOpenAI(
    base_url="http://192.168.98.39:1234/v1",
    api_key="lm-studio", 
    model="local-model", 
    temperature=0.1,      
)

# ==========================================
# 3. 核心大腦：RAG-Skill 漸進式檢索策略
# ==========================================
system_prompt = system_prompt = """你是一個【智慧照護系統的核心大腦】。你同時具備「控制實體家電」與「檢索本地知識庫」的能力。

【第一階段：意圖分流（🚨 極度重要 🚨）】
收到使用者的對話後，你【必須】先判斷這是哪一種需求，並選擇對應的行動：

👉 情境 A（環境控制/生理不適）：使用者抱怨環境（如「太亮了」、「太熱了」）或要求控制設備。
   ✅ 你的動作：【絕對禁止】去查知識庫！你必須立刻呼叫 `trigger_n8n_workflow` 工具，將動作指令（如 turn_off_light, close_curtain）傳送給 n8n 來控制設備。

👉 情境 B（查閱知識/設備手冊）：使用者明確詢問規定、操作手冊、注意事項等資訊。
   ✅ 你的動作：請啟動下方的【知識庫漸進式檢索流程】去找答案。

---
【第二階段：知識庫漸進式檢索流程 (僅限情境 B 啟用)】
1. 【分層導航】：永遠先從根目錄開始。如果不知道方向，呼叫 `list_directory` 傳入空字串 `""` 查看目前目錄結構。
2. 【強制閱讀索引】：任何目錄下只要有 `data_structure.md`，你【必須】先呼叫 `read_text_file` (或 read_file_progressively) 去閱讀它。
3. 【精準定位】：根據索引的說明，判斷使用者的問題應該去哪個子目錄。
4. 【局部讀取】：遇到 Markdown 讀取文字；遇到 PDF 預覽內容；遇到 Excel 讀取結構。
5. 【統整輸出】：用繁體中文回答。基於你探索到的真實內容回答，【絕對禁止】在沒有讀取檔案的情況下自己瞎掰。找不到請誠實說找不到。"""

# ==========================================
# 4. 建立並執行 Agent
# ==========================================
agent = create_react_agent(llm, tools, prompt=system_prompt)

def main():
    print("=== RAG-Skill 漸進式檢索 Agent 啟動 ===")
    
    # 檢查：確保路徑真的存在
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"\n🚨 警告：系統找不到你設定的知識庫目錄！")
        print(f"請確認這個路徑是否存在：{KNOWLEDGE_BASE_DIR}")
        return
    else:
        print(f"✅ 成功連接知識庫目錄：{KNOWLEDGE_BASE_DIR}\n")
    
    
    user_input = "天色有點暗了"
    
    print(f"👤 提問: {user_input}\n")
    
    result = agent.invoke(
        {"messages": [("user", user_input)]},
        config={"recursion_limit": 15} 
    )
    
    print("\n=== 🤖 最終回答 ===")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()