
import requests

URL = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3.5:4b",
    "messages": [
        {
            "role": "user",
            "content": (
                "Explain three common reasons why a car "
                "engine overheats. Use simple language."
            )
        }
    ],
    "think": False,
    "stream": False,
    "keep_alive": "5m",
    "options": {
        "temperature": 0.2,
        "num_predict": 256,
        "num_ctx": 2048
    }
}

try:
    print("Sending request to Qwen...", flush=True)

    response = requests.post(
        URL,
        json=payload,
        timeout=(10, 600)
    )

    response.raise_for_status()

    data = response.json()
    message = data.get("message", {})

    answer = message.get("content", "")
    thinking = message.get("thinking", "")

    print("\n" + "=" * 60)
    print("QWEN GENERATED ANSWER")
    print("=" * 60)

    if answer.strip():
        print(answer)
    else:
        print("No final answer was returned.")
        if thinking:
            print("\nThinking output was received:")
            print(thinking)
        print("\nResponse details:", data)

    print("\n" + "=" * 60)
    print("Generation time:", data.get("total_duration"), "ns")
    print("Generated tokens:", data.get("eval_count"))

except requests.exceptions.Timeout:
    print("Qwen timed out. The model may need more memory.")

except requests.exceptions.ConnectionError:
    print("Cannot connect to Ollama. Check whether it is running.")

except requests.exceptions.RequestException as error:
    print("Request error:", error)
