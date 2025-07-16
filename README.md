### 🔧 Setup and Installation

Follow these steps to set up the project on your local machine.

#### 1. Prerequisites

* **Python 3.8+**
* **Reddit API Credentials:** You'll need a Client ID and Client Secret. You can create a new "script" app on Reddit's [apps page](https://www.reddit.com/prefs/apps).
* **Groq API Key:** You can generate a free API key from the [Groq Console](https://console.groq.com/keys).

#### 2. Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Gagan2004/Reddit_Persona_Analyzer
    cd reddit-persona-analyzer
    ```

2.  **Install Dependencies**
     Then, install the packages using pip:

    ```
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create the Environment File**
    Create a file named `.env` in the root of the project directory and add your API keys. 

    ```env
    # .env file

    # Get these from [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
    REDDIT_CLIENT_ID="your_reddit_client_id"
    REDDIT_CLIENT_SECRET="your_reddit_client_secret"
    REDDIT_USER_AGENT="PersonaAnalyzer/1.0 by u/YourUsername"

    # Get this from [https://console.groq.com/keys](https://console.groq.com/keys)
    GROQ_API_KEY="your_groq_api_key"
    ```


    ## Command Structure
You run the script from your terminal or command prompt using the following structure:
 ```bash
python reddit_persona_analyzer.py <profile_url> [options]

```


### ## Arguments

| Argument      | Shorthand | Description                                                                             | Required |
| ------------- | --------- | --------------------------------------------------------------------------------------- | :------: |
| `profile_url` | *(none)* | **Required.** The full URL of the Reddit user's profile to analyze.                     | **Yes** |
| `--limit`     | `-l`      | Sets the maximum number of recent posts and comments to fetch.                          |    No    |
| `--format`    | *(none)* | Specifies the output format. Choices are `txt` (default) or `json`.                     |    No    |

---

### ## Examples

* **Basic Execution (generates a `.txt` report):**
    ```bash
    python reddit_persona_analyzer.py https://www.reddit.com/user/some_username/
    ```

* **Generate a JSON File for Data Analysis:**
    ```bash
    python reddit_persona_analyzer.py  https://www.reddit.com/user/some_username/ --format json
    ```

* **Analyze More Content (e.g., 250 items):**
    ```bash
    python reddit_persona_analyzer.py https://www.reddit.com/user/some_username/ --limit 250
    ```

* **Using the Shorthand for Limit:**
    ```bash
    python reddit_persona_analyzer.py https://www.reddit.com/user/some_username/ -l 50
    ```

---
### ## 📂 Output Files Explained

* **`persona_{username}.txt`**: A polished, formatted report designed for human reading. It includes a high-level summary, detailed traits with confidence scores, and supporting evidence.
* **`persona_{username}.json`**: A complete, structured data file containing the entire persona object, including all traits and their specific citations. This format is ideal for use in other applications or for data analysis pipelines.
* **`scraper.log`**: A persistent log file that records a timestamp and a success message every time a persona file is created or updated, providing an audit trail of the script's activity.

---


