import React from 'react';
import "./Navbar.css"
import image from "./logo5.png"
export const NavBar = () => {

    return(
    <nav className="navbar bg-dark fixed-top">
      <form className="container-fluid justify-content-start">
        <a className="nav-brand nav-link" aria-current="page" href="/"><strong>ImaginAI</strong></a>
        <button style={{marginLeft:'2%'}} className="btn btn-dark me-2" type="button"><a className="nav-link" aria-current="page" href="/formpage">Submit Scan</a></button>
      </form>
      <img style={{position:'absolute', marginLeft:'85%'}} width={'150px'} src={image} alt='none'></img>
    </nav>
    )
}

export default NavBar;