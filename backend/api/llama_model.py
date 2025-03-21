from huggingface_hub import login
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from django.db.models import Q
from api.models import Customer  # Import your Customer model
import os
from dotenv import load_dotenv

load_dotenv()



# ✅ Hugging Face API Token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_KEY")
login(HUGGINGFACE_TOKEN)

# ✅ Model Name
MODEL_NAME = "meta-llama/Llama-3.2-1B"

print("🔷 Loading LLaMA 3.2 Model on GPU...")

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Force GPU usage
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()  # ✅ Clears previous memory (optional)
else:
    raise RuntimeError("❌ No GPU detected! Please check your CUDA installation.")

dtype = torch.float16  # ✅ Use float16 for GPU efficiency
print(f"🔥 Using Device: {device.upper()}")
print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA Version: {torch.version.cuda}")
print(f"✅ PyTorch CUDA Available: {torch.cuda.is_available()}")

# ✅ Load model on GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map={"": 0},  # ✅ Force model to use GPU 0
)

# ✅ Compile model for faster execution
model = torch.compile(model)

print(f"✅ LLaMA 3.2 Model Loaded Successfully on {device.upper()} 🚀")


def fetch_churn_data():
    """Retrieve all customer churn data from the database."""
    try:
        customers = Customer.objects.all()
        churn_data = [
            {
                "Customer ID": c.customerID,
                "Churn": "Yes" if c.Churn else "No",
                "Tenure": c.tenure,
                "Satisfaction Score": c.SatisfactionScore,
                "Monthly Charges": c.MonthlyCharges,
                "Total Charges": c.TotalCharges,
                "Customer Support Calls": c.CustomerSupportCalls,
            }
            for c in customers
        ]
        return churn_data
    except Exception as e:
        print(f"Database Error: {e}")
        return None


def generate_response(user_message):
    """
    Generate chatbot response based on customer churn prediction data.
    """
    churn_data = fetch_churn_data()

    if not churn_data:
        return "I couldn't fetch customer churn data. Please check the database connection."

    # ✅ Check if user asks about churn or retention
    if "churn" in user_message.lower():
        churn_count = sum(1 for c in churn_data if c["Churn"] == "Yes")
        return f"There are {churn_count} customers likely to churn."

    if "retention" in user_message.lower():
        retained_count = sum(1 for c in churn_data if c["Churn"] == "No")
        return f"There are {retained_count} retained customers."

    # ✅ If no predefined answer, use LLaMA to generate a response
    input_text = f"<s>[INST] {user_message} [/INST]</s>"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # ✅ Generate `attention_mask`
    attention_mask = torch.ones_like(input_ids).to(device)

    # ✅ Set `pad_token_id`
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ✅ Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=300,
            pad_token_id=pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # ✅ Clean response
    response = response.replace("<s>", "").replace("</s>", "").replace("[INST]", "").replace("[/INST]", "").strip()

    return response
