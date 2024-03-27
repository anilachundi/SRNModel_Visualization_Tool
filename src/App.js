import React, { useState, useEffect , useRef } from 'react';
import * as d3 from 'd3';
import Header from './components/Header';
import Footer from './components/Footer';
import {Button,Button2} from './components/Button';
// import { createRoot } from 'react-dom/client';
// // import App from './App'; // Adjust the import path according to your file structure

// const container = document.getElementById('root'); // The DOM element where your app will be mounted
// const root = createRoot(container); // Create a root.

// root.render(<App />);

function App() {
  const [embeddingLayer, setEmbeddingLayer] = useState([]);
  const [hiddenLayer, setHiddenLayer] = useState([]);
  const [outputLayer, setOutputLayer] = useState([]);

  

  // State for active status of hidden layer buttons
  const [hiddenLayerActiveStatus, setHiddenLayerActiveStatus] = useState(Array(12).fill(true));

  const [leftActiveIndex, setLeftActiveIndex] = useState(null);
  // const [middleActiveIndex, setMiddleActiveIndex] = useState(null);
  const [extensionActiveIndex, setExtensionActiveIndex] = useState(null);
  const [rightActiveIndex, setRightActiveIndex] = useState(null);

  const handleLeftButtonClick = (index, event) => {
    console.log('Button clicked', index);
    setLeftActiveIndex(index);
    const buttonRect = event.target.getBoundingClientRect();
    const startPos = {
      x: buttonRect.left + buttonRect.width / 2, // Center X of the button
      y: buttonRect.top + window.scrollY + buttonRect.height / 2, // Center Y of the button, adjusted for scrolling
    };
    // console.log(startPos);
    drawLines(startPos);
  };

  const handleHiddenLayerButtonClick = (index) => {
    console.log('Before update:', hiddenLayerActiveStatus);
    setHiddenLayerActiveStatus(hiddenLayerActiveStatus.map((active, i) => i === index ? !active : active));
    console.log('After update:', hiddenLayerActiveStatus);
  };

  const handleRightButtonClick = (index) => {
    setRightActiveIndex(index);
    
  };

  const handleExtensionButtonClick = (index) => {
    setExtensionActiveIndex(index);

  }

  const extension_layer = Array.from({ length: 16 }, (_, index) => index + 1);


  const svgRef = useRef(null);

  const calculateEndPos = (hiddenIndex) => {
  // Constants for button layout - adjust these to match your actual layout
  const initialTopPosition = 95; // Example starting Y position for the first button in the hidden layer
  const buttonHeight = 5; // Example button height
  const verticalSpacing = 37; // Space between buttons
  const horizontalPosition = 850; // Example fixed X position for buttons in the hidden layer

  const yPos = initialTopPosition + hiddenIndex * (buttonHeight + verticalSpacing) + (buttonHeight / 2);
  
  const endPos = {
    x: horizontalPosition + (buttonHeight / 2), // Center X of the button
    y: yPos, // Center Y of the button
  };

  return endPos;
  };

  const drawLines = (startPos) => {
    console.log('Drawing lines from:', startPos);
    d3.select(svgRef.current).selectAll('line').remove(); // Clear existing lines
  
    hiddenLayer.forEach((_, hiddenIndex) => {
      console.log("hiddenIndex is: ", hiddenIndex);
      if (hiddenLayerActiveStatus[hiddenIndex]) { // Check if the button is active/connected
        console.log("entering if statement");
        const endPos = calculateEndPos(hiddenIndex); // Calculate the end position based on the index
        // Draw a line from startPos to endPos
        console.log('Drawing lines to:', endPos);
        d3.select(svgRef.current)
          .append('line')
          .attr('x1', startPos.x)
          .attr('y1', startPos.y)
          .attr('x2', endPos.x)
          .attr('y2', endPos.y)
          .attr('stroke', 'black')
          .attr('stroke-width', 2);
      }
    });
  };


  useEffect(() => {
    const fetchModelSummary = async () => {
      const response = await fetch('http://localhost:5000/model-summary', {
        method: 'GET', // Specifying the method as GET
      });

      if (response.ok) {
        const summary = await response.json();
        // Update the layers based on the fetched model summary
        setEmbeddingLayer(Array.from({ length: summary.embedding_layer.output_features }, (_, index) => index + 1));
        setHiddenLayer(Array.from({ length: summary.hidden_layer.output_features }, (_, index) => index + 1));
        setOutputLayer(Array.from({ length: summary.output_layer.output_features }, (_, index) => index + 1));
      }
    };

    fetchModelSummary();
  }, [hiddenLayerActiveStatus]);

  return (
    <div className="relative pb-10 min-h-screen">
      <Header />
      <svg ref={svgRef} className="absolute top-0 left-0 w-full h-full" style={{zIndex: 1, pointerEvents: 'none'}}></svg>
      
      <div className="flex relative">
        <div className="flex">
          <div className="flex flex-col">
            <div className="flex flex-col items-center h-full ml-20">
              <div className="text-center font-300 font-bold my-4">
                {"INPUT LAYER:"}
              </div>
              {embeddingLayer.map((buttonIndex, index) => (
                <div key={index} className="my-2">
                  <Button
                    isActive={leftActiveIndex === index}
                    // console.log('index before clicking: ', index);
                    // console.log('e is : ', e);
                    onClick={(e) => handleLeftButtonClick(index, e)}
                  />
                </div>
              ))}
            </div>
          </div>
          

          <div className="flex flex-col items-center h-full ml-20">
            <div className="text-center font-300 font-bold my-4"> {/* Add margin-top and margin-bottom for spacing */}
              {"PREVIOUS STATE:"}
            </div>
            {extension_layer.map((buttonIndex) => (
              <div key={buttonIndex} className="my-2">
                <Button2
                  text={`Button ${buttonIndex}`}
                  isActive={extensionActiveIndex === buttonIndex}
                  onClick={() => handleExtensionButtonClick(buttonIndex)}
                />
              </div>
            ))}
          </div>
        </div>
        

        
        <div className="flex flex-col items-center justify-center h-full mx-auto">
          <div className="text-center font-300 font-bold my-4">
            {"HIDDEN LAYER:"}
          </div>
          {hiddenLayer.map((_, index) => (
            <div key={index} className="my-2">
              <Button
                isActive={hiddenLayerActiveStatus[index]}
                onClick={() => handleHiddenLayerButtonClick(index)}
              />
            </div>
          ))}
        </div>

        <div className="flex flex-col items-center h-full mr-20">
          <div className="text-center font-300 font-bold my-4"> {/* Add margin-top and margin-bottom for spacing */}
          {"OUTPUT LAYER:"}
        </div>
          {outputLayer.map((buttonIndex) => (
            <div key={buttonIndex} className="my-2">
              <Button
                text={`Button ${buttonIndex}`}
                isActive={rightActiveIndex === buttonIndex}
                onClick={() => handleRightButtonClick(buttonIndex)}
              />
            </div>
          ))}
        </div>

      </div>

      <Footer />
    </div>
  );
}

export default App;

