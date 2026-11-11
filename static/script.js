// ---------------------------------------
// GET HTML ELEMENTS
// ---------------------------------------

const imageInput = document.getElementById("imageInput");

const dropArea = document.getElementById("dropArea");

const browseButton = document.getElementById("browseButton");

const uploadContent = document.getElementById("uploadContent");

const previewContainer = document.getElementById("previewContainer");

const previewImage = document.getElementById("previewImage");

const fileName = document.getElementById("fileName");

const removeImage = document.getElementById("removeImage");

const analyzeButton = document.getElementById("analyzeButton");

const loadingText = document.getElementById("loadingText");

const errorMessage = document.getElementById("errorMessage");

const emptyResult = document.getElementById("emptyResult");

const resultContent = document.getElementById("resultContent");

const gradcamSection = document.getElementById("gradcamSection");

const gradcamImage = document.getElementById("gradcamImage");

let selectedFile = null;


// ---------------------------------------
// SHOW ERROR
// ---------------------------------------

function showError(message) {

    errorMessage.textContent = message;

    errorMessage.classList.remove("hidden");

}


// ---------------------------------------
// OPEN FILE BROWSER
// ---------------------------------------

browseButton.addEventListener("click", function () {

    imageInput.click();

});


// ---------------------------------------
// IMAGE SELECTION
// ---------------------------------------

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (file) {

        handleFile(file);

    }

});


// ---------------------------------------
// HANDLE IMAGE
// ---------------------------------------

function handleFile(file) {

    errorMessage.classList.add("hidden");

    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {

        showError("Please select a JPG or PNG image.");

        return;

    }

    if (file.size > 10 * 1024 * 1024) {

        showError("Image size must be less than 10 MB.");

        return;

    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src = event.target.result;

        fileName.textContent = file.name;

        uploadContent.classList.add("hidden");

        previewContainer.classList.remove("hidden");

        analyzeButton.disabled = false;

        resultContent.classList.add("hidden");

        emptyResult.classList.remove("hidden");

        gradcamSection.classList.add("hidden");

    };

    reader.onerror = function () {

        showError("Could not read this image. Please try another file.");

    };

    reader.readAsDataURL(file);

}


// ---------------------------------------
// REMOVE IMAGE
// ---------------------------------------

removeImage.addEventListener("click", function () {

    selectedFile = null;

    imageInput.value = "";

    previewImage.src = "";

    previewContainer.classList.add("hidden");

    uploadContent.classList.remove("hidden");

    analyzeButton.disabled = true;

    resultContent.classList.add("hidden");

    emptyResult.classList.remove("hidden");

    gradcamSection.classList.add("hidden");

    errorMessage.classList.add("hidden");

});


// ---------------------------------------
// DRAG AND DROP
// ---------------------------------------

dropArea.addEventListener("dragover", function (event) {

    event.preventDefault();

    dropArea.classList.add("dragover");

});


dropArea.addEventListener("dragleave", function () {

    dropArea.classList.remove("dragover");

});


dropArea.addEventListener("drop", function (event) {

    event.preventDefault();

    dropArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (file) {

        handleFile(file);

    }

});


// ---------------------------------------
// ANALYZE IMAGE
// ---------------------------------------

analyzeButton.addEventListener("click", async function () {

    if (!selectedFile) {

        showError("Please upload an image first.");

        return;

    }

    analyzeButton.disabled = true;

    loadingText.classList.remove("hidden");

    errorMessage.classList.add("hidden");

    gradcamSection.classList.add("hidden");

    const formData = new FormData();

    formData.append("image", selectedFile);

    try {

        const response = await fetch("/predict", {

            method: "POST",

            body: formData

        });

        const responseText = await response.text();

        console.log("Server status:", response.status);

        console.log("Server response:", responseText);

        if (!responseText) {

            throw new Error(
                "The server returned an empty response. Status: " +
                response.status
            );

        }

        let data;

        try {

            data = JSON.parse(responseText);

        } catch (parseError) {

            throw new Error(
                "Invalid server response. Please check Render logs."
            );

        }

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Prediction failed. Status: " + response.status
            );

        }

        displayResults(data);

    } catch (error) {

        console.error("Prediction Error:", error);

        showError(
            error.message || "Something went wrong. Please try again."
        );

    } finally {

        analyzeButton.disabled = false;

        loadingText.classList.add("hidden");

    }

});


// ---------------------------------------
// DISPLAY RESULTS
// ---------------------------------------

function displayResults(data) {

    emptyResult.classList.add("hidden");

    resultContent.classList.remove("hidden");

    const prediction = data.prediction;

    const normalScore = Number(data.normal_score);

    const pneumoniaScore = Number(data.pneumonia_score);

    document.getElementById("predictionText").textContent =
        "MODEL CLASSIFICATION";

    const resultStatus = document.getElementById("resultStatus");

    const resultTitle = document.getElementById("resultTitle");

    const resultDescription = document.getElementById("resultDescription");

    resultTitle.textContent = prediction;

    resultDescription.textContent = data.description || "";

    if (prediction === "PNEUMONIA") {

        resultStatus.classList.add("pneumonia");

        resultTitle.style.color = "#d35e4e";

    } else {

        resultStatus.classList.remove("pneumonia");

        resultTitle.style.color = "#248b69";

    }


    // DISPLAY SCORES

    document.getElementById("normalScore").textContent =
        normalScore + "%";

    document.getElementById("pneumoniaScore").textContent =
        pneumoniaScore + "%";


    // DISPLAY PROGRESS BARS

    document.getElementById("normalBar").style.width =
        normalScore + "%";

    document.getElementById("pneumoniaBar").style.width =
        pneumoniaScore + "%";


    // DISPLAY GRAD-CAM

    if (data.gradcam) {

        gradcamImage.src =
            "data:image/png;base64," + data.gradcam;

        gradcamSection.classList.remove("hidden");

    } else {

        gradcamSection.classList.add("hidden");

        if (data.gradcam_message) {

            console.log(data.gradcam_message);

        }

    }


    // SCROLL TO RESULT

    resultContent.scrollIntoView({

        behavior: "smooth",

        block: "center"

    });

}
