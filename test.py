
import os
from dotenv import load_dotenv
from langchain_aws import BedrockLLM, BedrockEmbeddings
import numpy as np

# Load environment variables from .env file
load_dotenv()

print("AWS credentials from environment:")
print(f"Access Key ID: {os.environ.get('AWS_ACCESS_KEY_ID', 'Not found')[:5]}...")
print(f"Secret Key: {'*****' if os.environ.get('AWS_SECRET_ACCESS_KEY') else 'Not found'}")
print(f"Region: {os.environ.get('AWS_REGION', 'Not found')}")
print("-" * 50)

def test_bedrock_llm():
    print("\nTesting Bedrock LLM (Text Generation)...")
    try:
        # Initialize the Bedrock LLM
        llm = BedrockLLM(
            model_id="amazon.titan-text-premier-v1:0",  # Using Titan Express model
            region_name="us-east-1"
        )
        
        # Simple prompt
        prompt = "Write a short greeting in 10 words or less."
        print(f"Prompt: {prompt}")
        
        # Invoke the model
        result = llm.invoke(prompt)
        print(f"Result: {result}")
        print("✅ LLM Test Success!")
        return True
        
    except Exception as e:
        print(f"❌ LLM Test Failed: {str(e)}")
        return False

def test_bedrock_embeddings():
    print("\nTesting Bedrock Embeddings...")
    try:
        # Initialize the Bedrock Embeddings
        embed = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",  # Using Titan Embeddings V2
            region_name="us-east-1"
        )
        
        # Text to embed
        text = "This is a test sentence for embeddings."
        print(f"Text: {text}")
        
        # Get embeddings
        embedding = embed.embed_query(text)
        
        # Print embedding info (not the whole thing)
        embedding_array = np.array(embedding)
        print(f"Embedding shape: {embedding_array.shape}")
        print(f"First few values: {embedding_array[:5]}")
        print(f"Embedding norm: {np.linalg.norm(embedding_array)}")
        print("✅ Embeddings Test Success!")
        return True
        
    except Exception as e:
        print(f"❌ Embeddings Test Failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Bedrock API Tests")
    print("=" * 50)
    
    llm_success = test_bedrock_llm()
    embeddings_success = test_bedrock_embeddings()
    
    print("\nTest Summary:")
    print(f"LLM Test: {'✅ Passed' if llm_success else '❌ Failed'}")
    print(f"Embeddings Test: {'✅ Passed' if embeddings_success else '❌ Failed'}")
    
    if llm_success and embeddings_success:
        print("\n🎉 All tests passed! Your Bedrock configuration is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
