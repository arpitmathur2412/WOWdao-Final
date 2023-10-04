import React from 'react';
import { NavBar } from '../components/NavBar';
import imginput from '../images/mri1.jpg'
import imgexpected from '../images/ct1.jpg'
import imgoutput from '../images/predicted.png'
import "./home.css"

export const Home = () => {
    return (
        <>
            <NavBar /><br />
            <div className='container-fluid' style={{marginTop: '2em', paddingTop: '3em'}}>
                <div style={{marginTop:'2em'}} className="container text-center">
                    <div style={{marginTop:'2em'}} className="row">
                        <h3>Revolutionizing the healthcare sector with the power of Generative AI <br></br> in CT-MRI scan conversion</h3>
                    </div>
                    <div className="row text-center" style={{marginTop:'4em',flexFlow: 'row', flex: 'row', flexShrink: 'inherit'}}>
                        <div className="col col-lg-4 col-md-4 col-sm-4 col-xsm-4">
                            <h5 style={{marginTop:"2em"}}>Input Image</h5>
                            <img src={imginput} width={"70%"} style={{marginTop:"2vh" ,borderRadius: '10px'}} className='img img-fluid rounded' alt='...' />
                        </div>
                        <div className="col col-lg-4 col-md-4 col-sm-4 col-xsm-4">
                            <h5 style={{marginTop:"2em"}}>Expected Image</h5>
                            <img src={imgexpected} width={"70%"} style={{marginTop:"2vh", borderRadius: '10px'}} className='img img-fluid rounded' alt='...' />
                        </div>
                        <div className="col col-lg-4 col-md-4 col-sm-4 col-xsm-4">
                            <h5 style={{marginTop:"2em"}}>Output Image</h5>
                            <img src={imgoutput} width={"70%"} style={{marginTop:"2vh", borderRadius: '10px'}} className='img img-fluid rounded' alt='...' />
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default Home;
