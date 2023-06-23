from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import streamlit as st

db_username = st.secrets["db_username"]
db_pswd = st.secrets["db_pswd"]
cluster_name = st.secrets["cluster_name"]


def test_connection():
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    
def get_user_collection(client, collection_name, recreate=False):
    mydb = client["database"]
    collist = mydb.list_collection_names()
    if collection_name in collist and not recreate:
        print(f"The collection - {collection_name} - exists.")
        return mydb[collection_name]
    else:
        print(f"The collection - {collection_name} - has been created.")
        mydb[collection_name].drop()
        return mydb[collection_name]

def add_userdata(mycol, username, password):
    res = mycol.insert_one({username: password})
    print("Added user with id: ", res.inserted_id)

def login_user(mycol, username,password):
    res = mycol.find_one({username: password})
    return res

def view_all_docs(mycol, max_items=100):
    cursor = mycol.find({})
    for document in cursor[:max_items]:
        print(document)


if __name__ == "__main__":
    uri = f"mongodb+srv://{db_username}:{db_pswd}@{cluster_name}.gqraedl.mongodb.net/test"
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    # Test ping
    test_connection()
    # Test collection creation
    collection_name = "rlhf_test"
    mycol = create_user_collection(client, collection_name, new=True)
    # Test adding a username and password to the collection
    test_username = "rlhf_user"
    test_password = "rlhf_pw"
    add_userdata(mycol, test_username, test_password)
    # Test run the view all dosc
    view_all_docs(mycol)
    # Test logging in with the dummy user
    ## Test with an incorrect username and password
    incorrect_res = login_user(mycol, test_username, test_password + "123")
    ## Test with a correct username and password
    correct_res = login_user(mycol, test_username, test_password)
    # View the difference in response with correct / incorrect conditions
    print("Response from the incorrect credentials: ", incorrect_res)
    print("Response from the correct credentials: ", correct_res)
