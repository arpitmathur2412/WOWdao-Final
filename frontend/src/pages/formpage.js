import { NavBar } from "../components/NavBar";
import React, { useState } from "react";
import axios from "axios";

export const Formpage = () => {
  const [imageFile, setImageFile] = useState();
  const [selectedFileName, setSelectedFileName] = useState("");
  const [imageStatusMessage, setImageStatusMessage] = useState("Not Ready!");
  const [outputImageURL, setOutputImageURL] = useState("");
  const [imageConversionType, setImageConversionType] = useState("ct-mri"); //for type conversion identification

  let apiUrl = `http://localhost:5000/api/v1/predict?type=${imageConversionType}`; //api url for backend

  function handleChange(e) {
    setImageFile(URL.createObjectURL(e.target.files[0]));
    const selectedFile = e.target.files[0];
    setImageFile(selectedFile);
    setSelectedFileName(selectedFile ? selectedFile.name : "");
  }

  const handleChangeQueryParamTo_CTMRI = (e) => {
    //changes image conversion type to ct-mri
    setImageConversionType("ct-mri");
  }

  const handleChangeQueryParamTo_MRICT = (e) =>{
    //changes image conversion type to mri-ct
    setImageConversionType("mri-ct");
  }

  const handleSubmit = async (e) => {
    console.log("Inside the handling submit button")
    e.preventDefault();
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("type", "ct-mri"); 
    console.log(formData);

    // The API Url based upon the radio buttons of different conversions type
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    // The API URl for CT-MRI Conversion is given below
    
    // const apiUrl = "http://localhost:5000/api/v1/predict?type=ct-mri";

    // For MRI-CT Conversion Please use the following link
    // const apiUrl = "http://localhost:5000/api/v1/predict?type=mri-ct";

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    try {
  const response = await axios.post(apiUrl, formData);
  console.log(response);
  
  if (response.data && response.data.blob_url) {
    const imageUrl = response.data.blob_url;
    console.log(imageUrl);

    // Use fetch to ping the URL and trigger the download
    fetch(imageUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Network Error...Please connect to the internet and try again');
        }
        return response.blob();
      })
      .then((blob) => {
        // Handle the blob data (e.g., display or process it)
        setImageStatusMessage("Ready!");
        const url = URL.createObjectURL(blob);
        setOutputImageURL(url);
      })
      .catch((error) => {
        console.error('Fetch error:', error);
      });
  } else {
    console.error("Invalid URL in the response");
  }
}   catch (error) {
      console.error(error);
    }
  };

  return (
    <>
        <NavBar />
        <div className="container-fluid d-flex justify-content-center align-items-center" style={{color:'white',minHeight: '130vh' , marginLeft: '10%vw', marginRight: '10%vw', justifyContent: 'space-around'}}>
              <div className="row text-center" style={{marginTop:'4em',flexFlow: 'row', flex: 'row', flexShrink: 'inherit'}}>
                    <div className="col col-lg-4 col-md-4 col-sm-4 col-xsm-4" style={{ borderColor: 'white', borderStyle: 'solid', borderWidth: '2px', paddingTop: '4px',borderRadius: '15px', width: "250px", height: "300px", marginRight: '30px'}}>
                      <form onSubmit={handleSubmit}>
                        {//try removing the on submit function from form temporarily to see if this works by calling the button based functions
                        }
                        <h5><blockquote style={{ paddingTop: '4px'}}>Input Scan Image</blockquote></h5><br />
                        <div className="form-group">
                          <input className="btn btn-light"
                            type="file"
                            onChange={handleChange}
                            
                            id="imageFormControlInput1"
                            required={true} style={{textAlign: 'center', width: '228px',
                            color:'white',
                            backgroundColor: 'black'}}
                          />
                        </div>
                        {/* Display the selected filename */}
          {selectedFileName && (
            <p>Selected File: {selectedFileName}</p>
          )}
                        <br />
                        <button type="submit" onClick={handleChangeQueryParamTo_CTMRI} className="btn btn-outline-success" style={{marginRight: '4%'}}>
                          CT-MRI
                        </button>
                        <button type="submit" onClick={handleChangeQueryParamTo_MRICT} className="btn btn-outline-success">
                          MRI-CT
                        </button>
                      </form>
                    </div>
                    <div className="col col-lg-4 col-md-4 col-sm-4 col-xsm-4" style={{ borderColor: 'white', borderStyle: 'solid', borderWidth: '2px',  paddingTop: '4px', borderRadius: '15px', width: "250px", height: "300px", marginLeft: '30px'}}>
                      <h5><blockquote style={{ paddingTop: '4px'}}>Output Image</blockquote></h5><br />
                      <form>
                        <p>{imageStatusMessage}</p><br/>
                        <a className="btn btn-outline-success" href={outputImageURL} target="_blank" rel="noopener noreferrer">Download</a>
                      </form>
                    </div>
              </div>
        </div>
    </>
  );
};

export default Formpage;
