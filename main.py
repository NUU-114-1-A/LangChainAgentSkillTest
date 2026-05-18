import os
import re 
import pandas as pd
import requests
import threading 
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import json
# 🚀 新增 Flask 相關套件
from flask import Flask, request, jsonify

# ==========================================
# 0. 設定目錄結構的絕對路徑
# ==========================================
KNOWLEDGE_BASE_DIR = r"C:\rag-skill\rag-skill\knowledge"

# ==========================================
# 1. 定義適配該結構的真實工具 (Tools)
# ==========================================


MEMORY_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "memories")
if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

@tool
def update_user_memory(user_id: str, new_info: str) -> str:
    """
    【專屬記憶管理】更新或記錄使用者的「關鍵記憶」。
    當 AI 發現使用者重要的生活習慣、病史、偏好或特殊需求時，必須呼叫此工具。
    user_id: 使用者的唯一識別碼。
    new_info: 要記錄的新資訊內容。
    """
    file_path = os.path.join(MEMORY_DIR, f"{user_id}.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"### 記錄時間：{timestamp}\n- {new_info}\n\n"
    
    # 🚀 加上這行！讓你的終端機印出大腦正在寫筆記的動作
    print(f"\n將記憶寫入: {file_path}")
    print(f"記憶內容: {new_info}")
    
    try:
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write(f"# 使用者 {user_id} 的專屬關鍵記憶\n\n")
            f.write(entry)
            
        print(f"[系統底層] 寫入成功！")
        return f"已成功將資訊寫入使用者 {user_id} 的記憶檔中。"
    except Exception as e:
        print(f"[系統底層] 寫入失敗: {str(e)}")
        return f"寫入記憶檔案發生錯誤：{str(e)}"
from datetime import datetime
@tool
def get_current_time() -> str:
    """
    獲取目前系統真實的日期與時間。
    當需要判斷現在是白天還是晚上、確認警報發生的時間點，或是使用者詢問時間時，請呼叫此工具。
    """
    now = datetime.now()
    # 將星期轉換為中文，讓 AI 更好理解
    weekday_map = {0: "星期一", 1: "星期二", 2: "星期三", 3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日"}
    weekday = weekday_map[now.weekday()]
    
    # 格式化輸出：2026-04-27 17:45:00 (星期一)
    time_str = now.strftime(f"%Y-%m-%d %H:%M:%S ({weekday})")
    
    return f"目前的系統時間是：{time_str}"
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
        return f"目錄 '{relative_path}' 下的內容：\n" + "\n".join(items)
    except FileNotFoundError:
        return f"錯誤：找不到目錄 '{relative_path}'"
    except Exception as e:
        return f"讀取目錄發生錯誤：{str(e)}"



@tool
def control_smart_device(device_key: str, action: str) -> str:
    """
    【系統強制要求】控制實體智慧家電 (例如：開關燈) 專用工具。
    當使用者說「天黑了」、「開燈」、「關燈」時，請優先呼叫此工具。
    device_key: 設備代號 (目前可用: "A" 代表 Lab_Plug)
    action: 欲執行的動作，必須是 "on" (開啟) 或 "off" (關閉)
    """
    if device_key not in DEVICE:
        return f"錯誤: 找不到代號 '{device_key}' 的設備。目前只有 'A'。"
        
    info = DEVICE[device_key]
    try:
        print(f"\n系統正在透過區域網路直接連線 Tuya 設備: {info['name']} ...")
        d = tinytuya.OutletDevice(info['id'], info['ip'], info['key'])
        d.set_version(3.4)
        
        if action.lower() == 'on':
            d.turn_on()
            status_text = "開啟"
        else:
            d.turn_off()
            status_text = "關閉"
            
        data = d.status()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{now}] 設備: {info['name']} | 狀態: {status_text}"
        print(f"系統：{log_msg}")
        
        return f"成功！已經將 {info['name']} {status_text}。"
        
    except Exception as e:
        return f"控制設備發生錯誤: {str(e)}。"

# ==========================================
# 同學的中繼網頁 API 設定
# ==========================================
API_BASE_URL = "http://192.168.98.46:5000" 

@tool
def read_sensor_data(sensor_type: str) -> str:
    """
    從中繼網頁 API 讀取最新的感測器數值。
    當使用者詢問環境狀態時呼叫此工具。
    sensor_type: 感測器種類。必須且只能是以下三者之一：
    - 'temperature' (查詢溫度)
    - 'humidity' (查詢濕度)
    - 'all' (未指定或同時查詢多項數值時使用)
    """
    try:
        url = f"{API_BASE_URL}/api/get_sensor"
        params = {"type": sensor_type}
        
        print(f"\nAPI請求 {sensor_type} 數據...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return f"最新 {sensor_type} 數值為: {data.get('value')}, 記錄時間: {data.get('timestamp')}"
        else:
            return f"無法取得資料，伺服器回應狀態碼: {response.status_code}"
    except Exception as e:
        return f"呼叫感測器 API 發生錯誤: {str(e)}"

@tool
def log_action_to_db(user_input: str, action_desc: str) -> str:
    """
    【系統強制要求】無論使用者問什麼、你做了什麼回答或控制，在最後回覆使用者「之前」，【絕對必須】呼叫此工具留下對話與處置紀錄。
    user_input: 使用者原始的提問或觸發條件 
    action_desc: 你的回覆摘要、執行的動作描述或給出的建議 (例如: "回答環境溫濕度"、"開啟電燈"、"提供失智症衛教")
    """
    try:
        url = f"{API_BASE_URL}/api/save_log"
        payload = {
            "user_input": user_input,
            "action_description": action_desc
        }
        
        print(f"\n系統正在將紀錄傳送給資料庫...")
        response = requests.post(url, json=payload)
        
        if response.status_code in [200, 201]:
            return "成功將「使用者輸入」與「處置紀錄」傳送至中繼網頁。"
        else:
            return f"傳送失敗，伺服器回應狀態碼: {response.status_code}"
    except Exception as e:
        return f"呼叫紀錄 API 發生錯誤: {str(e)}"

import os

@tool
def read_text_file(relative_file_path: str, num_lines: int = 100) -> str:
    """
    讀取 Markdown (.md) 或純文字檔的內容。
    【特別重要】：到達任何新目錄時，必須優先用此工具讀取該目錄下的 `data_structure.md`！
    傳入相對路徑 (例如 "data_structure.md" 或 "Safety Knowledge/data_structure.md")。
    """
    target_path = os.path.join(KNOWLEDGE_BASE_DIR, relative_file_path)
    
    # 🚀 實況轉播 1：告訴你大腦準備打開哪本書
    print(f"\n正在翻閱知識庫：尋找並打開檔案 '{relative_file_path}'...")
    
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
        
        print(f"翻閱失敗：找不到檔案或無法解析編碼 '{relative_file_path}'")
        return f"無法以任何已知編碼讀取檔案 '{relative_file_path}'。"

    if len(content) <= num_lines:
        
        print(f"讀取完畢 '{relative_file_path}' (共 {len(content)} 行內容)")
        return f"📄 {relative_file_path} 的完整內容：\n" + "".join(content)
    else:
        
        print(f"讀取完畢 已擷取 '{relative_file_path}' 的前 {num_lines} 行重點")
        return f"📄 {relative_file_path} 的前 {num_lines} 行內容：\n" + "".join(content[:num_lines])

@tool
def read_pdf_preview(relative_file_path: str, page_num: int = 0) -> str:
    """
    讀取 PDF 檔案的特定頁面內容 (預設讀取第 0 頁，也就是第一頁)。
    """
    target_path = os.path.join(KNOWLEDGE_BASE_DIR, relative_file_path)
    try:
        reader = PdfReader(target_path)
        if page_num >= len(reader.pages):
            return f"錯誤：該 PDF 只有 {len(reader.pages)} 頁。"
        
        page = reader.pages[page_num]
        text = page.extract_text()
        return f"📑 {relative_file_path} (第 {page_num} 頁) 內容擷取：\n{text[:1500]}..." 
    except Exception as e:
        return f"讀取 PDF 發生錯誤：{str(e)}"

@tool
def read_excel_preview(relative_file_path: str, num_rows: int = 5) -> str:
    """
    分析 Excel (.xlsx) 結構化資料。
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
        return f"讀取 Excel 發生錯誤：{str(e)}"

import json
@tool
def get_sensor_history_api_tool(hours: int = 6) -> str:
    """
    當你需要查詢長輩過去一段時間（例如：昨晚、過去幾小時）的歷史感測趨勢時，請呼叫此工具。
    這可以用來分析「睡眠品質」、「判斷異常是一時的還是持續的」，或是了解「環境溫濕度的變化趨勢」。
    輸入參數 hours 為整數，代表要往前查詢幾小時的資料（預設為 6，最多 24）。
    """
    # 將所有需要的套件直接 import 在函式內部，避免框架作用域抓不到的問題
    import os
    import json
    import requests


    print(f"\n[Agent Tool 觸發] 準備透過 API 查詢過去 {hours} 小時歷史趨勢...")
    print(f"目標 URL: {API_BASE_URL}/api/get_combined_history")
    
    try:
        # 透過 GET 請求呼叫 API，並將 hours 作為 query 參數傳遞
        response = requests.get(
            f"{API_BASE_URL}/api/get_combined_history", 
            params={"hours": hours}, 
            timeout=10
        )
        
        # 檢查 HTTP 狀態碼是否為 200 OK
        response.raise_for_status() 
        result = response.json()
        
        # 將回傳的 JSON 字典轉成字串，讓 AI 大腦可以閱讀
        return json.dumps(result, ensure_ascii=False)
        
    except requests.exceptions.ConnectionError:
        return f"系統回報：連線失敗！無法連接到歷史資料 API ({API_BASE_URL})，請確認 Flask 伺服器是否開啟。"
    except requests.exceptions.Timeout:
        return f"系統回報：API 請求超時！歷史資料庫伺服器沒有回應。"
    except requests.exceptions.RequestException as e:
        return f"系統回報：呼叫歷史資料 API 時發生未知錯誤：{str(e)}"
@tool
def push_message_api_tool(text: str, user_id: str = "U1e5c0bf175be6f76d7a861c75de412cd") -> str:
    """
    這是用來將訊息主動推播到家屬 LINE 上的工具。
    當你判斷有任何需要通知家屬的狀況（包含：健康異常警示、晨間健康總結報告、一般狀態回報，或是「系統/使用者要求測試推播」時），都可以隨時呼叫此工具。
    請將你想告訴家屬的完整內容傳入 text 參數中。
    """
    print(f"\n[Agent Tool 觸發] 準備透過 API 呼叫 LINE 推播...")
    print(f"目標 URL: {API_BASE_URL}/api/push_message")
    
    # 準備符合 API 規格的 JSON Payload
    payload = {
        "user_id": user_id,
        "text": text
    }
    
    try:
        # 發送 POST 請求給 Flask API (設定 timeout 避免 Agent 卡死)
        response = requests.post(f"{API_BASE_URL}/api/push_message", json=payload, timeout=10)
        
        # 檢查 HTTP 狀態碼 (200 OK)
        response.raise_for_status() 
        
        # 解析 Flask 回傳的 JSON
        result = response.json()
        
        if result.get("status") == "success":
            return f"系統回報：已成功透過 API 將緊急訊息推播給家屬。"
        else:
            return f"系統回報：API 呼叫成功，但推播失敗。伺服器訊息：{result.get('message')}"
            
    except requests.exceptions.ConnectionError:
        return f"系統回報：連線失敗！無法連接到推播 API ({API_BASE_URL})，請檢查伺服器是否啟動。"
    except requests.exceptions.Timeout:
        return f"系統回報：API 請求超時！推播伺服器沒有回應。"
    except requests.exceptions.RequestException as e:
        return f"系統回報：呼叫推播 API 時發生未知的網路錯誤：{str(e)}"
       
tools = [list_directory, 
         read_text_file, 
         read_pdf_preview, 
         read_excel_preview, 
         control_smart_device, 
         read_sensor_data, 
         log_action_to_db,
         get_current_time,
         get_sensor_history_api_tool,
         update_user_memory,
         push_message_api_tool]

# ==========================================
# 2. 連接 LM Studio 
# ==========================================
llm = ChatOpenAI(
    base_url="http://192.168.98.39:1234/v1/",  
    api_key="lm-studio", 
 
    temperature=0.1,      
    max_tokens=1000,             
    timeout=120                  
)

# ==========================================
# 3. 核心大腦：RAG-Skill 漸進式檢索策略
# ==========================================
system_prompt = """你是一個【智慧照護系統的核心大腦】。你同時具備「控制實體家電」、「檢索本地知識庫」、「讀取感測器」與「管理專屬記憶」的能力。

【第零階段：專屬記憶提取與管理】
1. 提取記憶：在制定照護決策時，請優先結合上下文記憶，若需要更詳盡的個人資訊，可嘗試呼叫 `read_text_file` 探索 `memories` 目錄。
2. 寫入記憶：當你在對話中發現以下高價值資訊時，請【主動呼叫 `update_user_memory`】工具將其永久記錄：
   - 長輩的特殊身體狀況或病史（如：對某藥物過敏、有高血壓、近期容易頭暈）。
   - 生活習慣與個人偏好（如：阿公下午三點習慣喝茶、睡覺必須留小燈）。
   - 家屬的特別叮嚀與照護要求。

【第一階段：意圖分流與複合行動】
收到使用者的對話後，請判斷需求並執行對應行動：
情境 A（環境控制/生理不適）：立刻呼叫 `control_smart_device` 或 `trigger_n8n_workflow`。
情境 B（查閱知識）：啟動【知識庫檢索流程】去找衛教答案。
情境 C（環境數據/歷史趨勢）：呼叫 `read_sensor_data` 或 `get_sensor_history` 讀取最新狀態與過去趨勢。
情境 D（混合需求與個人化）：先查閱專屬記憶與知識庫，結合感測數據，若需控制則連動家電，最後給出專屬建議。

【第二階段：知識庫漸進式檢索流程】
1. 永遠先從根目錄開始，可呼叫 `list_directory` 查看。
2. 看到 `data_structure.md` 必須優先呼叫 `read_text_file` 閱讀。
3. 基於探索到的真實內容回答，絕對禁止自己瞎掰。

---
【第三階段：全面強制紀錄（絕對嚴格遵守）】
無論使用者詢問什麼（就算是閒聊、查天氣、更新記憶、控制家電），在你得出最終答案並準備回覆使用者「之前」，你【絕對必須】先呼叫 `log_action_to_db` 將這次的互動記錄下來。
- user_input: 填入使用者的原始問題。
- action_desc: 填入你這次處置的精簡摘要（例如：「回答溫濕度趨勢」、「開啟電燈A防跌」、「更新阿公愛喝茶的個人記憶」、「提供失智症衛教資訊」）。
成功呼叫紀錄工具後，才能輸出最終答案給使用者。

---
【第四階段：輸出格式要求（絕對嚴格遵守）】
你的最終回答將直接傳送至 LINE Bot 與一般網頁介面。
1. 絕對禁止使用任何 Markdown 語法（包含但不限於 **粗體**、*斜體*、# 標題、`程式碼區塊`）。
2. 請使用純文字排版，段落之間直接使用換行。
3. 清單請使用簡單的數字 (1. 2. 3.) 或簡單符號 (如 - 或 •)。
4. 除非有特定要求，否則一律回答繁體中文。
"""
# ==========================================
# 4. 建立並執行 Agent
# ==========================================
memory = MemorySaver()
agent = create_react_agent(llm, tools, prompt=system_prompt, checkpointer=memory)

# 🚀 建立 Flask 應用程式
app = Flask(__name__)

@app.route("/api/ask_ai", methods=['POST'])
def ask_ai():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"status": "error", "reply": "缺少必要的 text 欄位"}), 400
            
        user_id = data.get("user_id", "default_user_session")
        user_input = data.get("text")
        
        print(f"\n[收到 API 請求] 來自使用者 {user_id}: {user_input}")
        
        config = {
            "configurable": {"thread_id": user_id}, 
            "recursion_limit": 15
        }
        
     
        combined_prompt = f"[系統資訊：當前對話使用者 ID 為 {user_id}]\n使用者說：{user_input}"
        
        # 呼叫大腦思考
        result = agent.invoke(
            {"messages": [("user", combined_prompt)]}, # 🚀 這裡改傳送 combined_prompt
            config=config 
        )
        
        
        raw_response = result["messages"][-1].content
        clean_response = raw_response
        clean_response = re.sub(r'\*{2}', '', clean_response)
        clean_response = re.sub(r'^\s*[\*\-]\s+', '• ', clean_response, flags=re.MULTILINE)
        clean_response = re.sub(r'^#+\s*', '', clean_response, flags=re.MULTILINE)
        
        print(f"收到回覆: {clean_response[:50]}...")
        
        return jsonify({
            "status": "success", 
            "reply": clean_response 
        }), 200

    except Exception as e:
        print(f"\n執行過程中發生錯誤：{str(e)}")
        return jsonify({
            "status": "error", 
            "reply": "發生錯誤，請稍後再試！"
        }), 500

#  新增：啟動 Flask 伺服器的執行緒函數
def run_flask_server():
    # 注意：在 Thread 裡面執行 Flask 時，use_reloader 必須設為 False
    app.run(host="0.0.0.0", port=8000, use_reloader=False)

if __name__ == "__main__":
    print("=== 智慧照護系統 AI 已啟動 ===")
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"\n找不到知識庫目錄({KNOWLEDGE_BASE_DIR})")
    else:
        print(f"成功連接知識庫目錄：{KNOWLEDGE_BASE_DIR}")
        
    print("="*50)
    print("啟動API 伺服器 (Port 8000)...")
    
    # 將 Flask 伺服器放到背景執行緒運作
    server_thread = threading.Thread(target=run_flask_server)
    server_thread.daemon = True # 設定為守護執行緒，主程式關閉時它也會自動關閉
    server_thread.start()
    
    print("(輸入 'q' 或 'quit' 即可離開程式)\n")
    print("="*50)
    
    
    while True:
        try:
            user_text = input("\n測試輸入: ")
            
            # 檢查是否要離開程式
            if user_text.lower() in ['q', 'quit', 'exit']:
                print("系統關閉中...")
                break
                
            # 避免輸入空白
            if not user_text.strip():
                continue
                
            print("\n思考中，請稍候...")
            
            # 手動測試時給予一個專用的 thread_id
            config = {
                "configurable": {"thread_id": "terminal_tester"}, 
                "recursion_limit": 15
            }
            
            # 呼叫 Agent
            result = agent.invoke(
                {"messages": [("user", user_text)]},
                config=config 
            )
            
            raw_response = result["messages"][-1].content
            
            # 文字淨化處理 (跟 API 一樣)
            clean_response = raw_response
            clean_response = re.sub(r'\*{2}', '', clean_response)
            clean_response = re.sub(r'^\s*[\*\-]\s+', '• ', clean_response, flags=re.MULTILINE)
            clean_response = re.sub(r'^#+\s*', '', clean_response, flags=re.MULTILINE)
            
            print(f"\nAI 回覆:\n{clean_response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            # 捕捉 Ctrl+C
            print("\n系統關閉中...")
            break
        except Exception as e:
            print(f"\n手動測試發生錯誤: {str(e)}")