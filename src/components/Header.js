import React from 'react'
// import Navigation from './Navigation'
// import {Link} from 'react-router-dom'
import '../App.css';


function Header(){
    return (

        <header className="border-b p-7 flex justify-center items-center">
            <p className='header-title'>
                SRN Model
            </p>

            {/* <Navigation /> */}
        </header>
    )
}

export default Header;