### 🔧 Setup and Installation

Follow these steps to set up the project on your local machine.

#### 1. Prerequisites

* **Python 3.8+**
* **Reddit API Credentials:** You'll need a Client ID and Client Secret. You can create a new "script" app on Reddit's [apps page](https://www.reddit.com/prefs/apps).
* **Groq API Key:** You can generate a free API key from the [Groq Console](https://console.groq.com/keys).

#### 2. Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Gagan2004/Reddit_Persona_Analyzer]
    cd reddit-persona-analyzer
    ```

2.  **Install Dependencies**
     Then, install the packages using pip:

    ```
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create the Environment File**
    Create a file named `.env` in the root of the project directory and add your API keys. **Do not share this file publicly.**

    ```env
    # .env file

    # Get these from [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
    REDDIT_CLIENT_ID="your_reddit_client_id"
    REDDIT_CLIENT_SECRET="your_reddit_client_secret"
    REDDIT_USER_AGENT="PersonaAnalyzer/1.0 by u/YourUsername"

    # Get this from [https://console.groq.com/keys](https://console.groq.com/keys)
    GROQ_API_KEY="your_groq_api_key"
    ```
