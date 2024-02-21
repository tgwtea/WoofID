import streamlit as st
import torch
import os
from torchvision import models, transforms
from PIL import Image
import io
import matplotlib.pyplot as plt

# Load your trained model
def load_model(model_path='best_woof_id_model.pth'):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 120)  # Adjust based on your dataset
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get breed names mapping
def get_breed_names(dataset_path='images'):
    breeds = sorted(os.listdir(dataset_path))  # Sort to ensure consistent index assignment
    return breeds

breed_names = get_breed_names()

# Prettify breed names
def prettify_breed_name(breed_name):
    # Replace underscores and hyphens with spaces
    pretty_name = breed_name.replace("_", " ").replace("-", " ")
    # Remove any digits and extra spaces
    pretty_name = ''.join([i for i in pretty_name if not i.isdigit()]).strip()
    # Capitalize each word
    pretty_name = ' '.join(word.capitalize() for word in pretty_name.split())
    return pretty_name


# Predict function with prettified breed names
def predict_breed(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img_t = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_t)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
        _, predicted_idx = torch.max(output, 1)

    # Map the predicted index to prettified breed name
    breed = prettify_breed_name(breed_names[predicted_idx.item()])
    probabilities = {prettify_breed_name(breed_names[i]): prob.item() for i, prob in enumerate(probabilities)}
    # Sort probabilities by highest first
    probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    return breed, probabilities

# Initialize Streamlit app
st.title('Woof ID 🐶', anchor=None)

uploaded_image = st.file_uploader("Upload an image of a dog", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Call model prediction function
    breed, probabilities = predict_breed(uploaded_image.getvalue())

    # Display predicted breed
    st.write(f"**Predicted Breed:** {breed}")

    # Display a pie chart for the top 3 predictions
    top_breeds = list(probabilities.items())[:3]
    fig, ax = plt.subplots()
    ax.pie([x[1] for x in top_breeds], labels=[x[0] for x in top_breeds], autopct='%1.1f%%')
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    st.pyplot(fig)

    # Display probabilities in a nicer format
    st.write("**Probabilities:**")
    for breed, probability in probabilities.items():
        st.write(f"{breed}: {probability:.2f}%")

    # Generate balloons/confetti
    st.balloons()
