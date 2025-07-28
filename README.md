# CodeLen - Repository Analysis and Chat Application

CodeLen is a full-stack application that provides repository analysis and AI-powered chat functionality for understanding codebases.

## Project Structure

```
CodeLen/
├── backend/                 # Python FastAPI backend
│   ├── main.py             # Main FastAPI application
│   ├── basic_rag.py        # RAG (Retrieval-Augmented Generation) implementation
│   ├── requirements.txt    # Python dependencies
│   ├── api.env            # Environment variables (template)
│   └── api.env.template   # Template for environment variables
├── reporangers/           # React frontend application
│   ├── src/
│   │   ├── Components/    # React components
│   │   └── ...
│   ├── public/
│   └── package.json
└── README.md
```

## Features

- **Repository Analysis**: Analyze GitHub repositories for code structure and metrics
- **AI Chat Interface**: Chat with AI about your codebase using RAG
- **Code Statistics**: Visual analytics and statistics about repositories
- **Multi-language Support**: Supports Python, TypeScript, JavaScript, and Rust

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `api.env.template` to `api.env`
   - Fill in your actual API keys and tokens:
     ```bash
     cp api.env.template api.env
     ```
   - Edit `api.env` with your credentials:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `GITHUB_TOKEN`: Your GitHub personal access token
     - `COHERE_API_KEYS`: Comma-separated Cohere API keys
     - `FIREBASE_CREDENTIAL_JSON`: Firebase service account credentials
     - `HUGGINGFACE_API_TOKEN`: Hugging Face API token

4. Run the backend server:
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd reporangers
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   The application will be available at `http://localhost:3000`

## API Endpoints

- `GET /` - Welcome message
- `POST /extract-repo` - Extract and analyze repository
- `POST /chat` - Chat with AI about the codebase
- Additional endpoints for analytics and statistics

## Technologies Used

### Backend
- **FastAPI**: Web framework for building APIs
- **LangChain**: Framework for LLM applications
- **Cohere**: AI language models
- **Firebase**: Database and authentication
- **ChromaDB**: Vector database for embeddings
- **Beautiful Soup**: Web scraping

### Frontend
- **React**: Frontend framework
- **Chart.js**: Data visualization
- **CSS**: Styling

## Security Notes

- Never commit actual API keys or sensitive credentials
- Use the provided template files and add your credentials locally
- The `.gitignore` file is configured to exclude sensitive files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure no sensitive data is committed
5. Submit a pull request

## License

This project is for educational purposes. Please ensure you have the proper licenses for all APIs and services used.