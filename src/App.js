import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import Header from './components/Header';
import { Button, Button2 } from './components/Button';
import './App.css';

function App() {
  const [embeddingLayer, setEmbeddingLayer] = useState([]);
  const [hiddenLayer, setHiddenLayer] = useState([]);
  const [outputLayer, setOutputLayer] = useState([]);
  const [embeddingLayerActiveStatus, setEmbeddingLayerActiveStatus] = useState(Array(17).fill(false));
  const [hiddenLayerActiveStatus, setHiddenLayerActiveStatus] = useState(Array(12).fill(false));
  const [rightActiveIndex, setRightActiveIndex] = useState(null);
  const [extensionActiveIndex, setExtensionActiveIndex] = useState(Array(12).fill(false)); // Update to an array for multiple states
  const [positions, setPositions] = useState([]);
  const svgRef = useRef(null);
  const prevHiddenLayerActiveStatusRef = useRef(hiddenLayerActiveStatus);

  const calculatePositions = useCallback((activeStatus, layerIdPrefix) => {
    const positions = [];
    activeStatus.forEach((isActive, index) => {
      if (isActive) {
        const buttonId = `${layerIdPrefix}-${index}`;
        const buttonElement = document.getElementById(buttonId);
        if (buttonElement) {
          const buttonRect = buttonElement.getBoundingClientRect();
          // console.log("buttonRect: ", buttonRect);
          positions.push({
            x: buttonRect.left + buttonRect.width / 2,
            y: buttonRect.top + window.scrollY + buttonRect.height / 2,
          });
        }
      }
    });
    return positions;
  }, []);

  const handleEmbeddingLayerButtonClick = (index) => {
    setEmbeddingLayerActiveStatus(prevState => {
      const newState = prevState.map((active, i) => i === index ? !active : active);
      const newPositions = calculatePositions(newState, 'button-embedding');
      setPositions(newPositions);
      return newState;
    });
  };

  const handleHiddenLayerButtonClick = (index) => {
    // Set the previous state before updating the current state
    setExtensionActiveIndex(prevHiddenLayerActiveStatusRef.current);
    setHiddenLayerActiveStatus(prevState => {
      const newState = prevState.map((active, i) => i === index ? !active : active);
      // Update the ref to the new state
      prevHiddenLayerActiveStatusRef.current = newState;
      return newState;
    });
  };

  const handleRightButtonClick = (index) => {
    setRightActiveIndex(index);
  };

  const handleExtensionButtonClick = (index) => {
    setExtensionActiveIndex(prevState => {
      return prevState.map((active, i) => i === index ? !active : active);
    });
  };

  const calculateEndPos = useCallback((hiddenIndex) => {
    
    const buttonId = `${'button-hidden'}-${hiddenIndex}`;
    const buttonElement = document.getElementById(buttonId);
    
    const buttonRect = buttonElement.getBoundingClientRect();

    return {
      x: buttonRect.left + buttonRect.width / 2,
      y: buttonRect.top + window.scrollY + buttonRect.height / 2,
    };
  }, []);

  const calculateOutputEndPos = useCallback((outputIndex) => {

    const buttonId = `${'button-output'}-${outputIndex}`;
    const buttonElement = document.getElementById(buttonId);
    
    const buttonRect = buttonElement.getBoundingClientRect();

    return {
      x: buttonRect.left + buttonRect.width / 2,
      y: buttonRect.top + window.scrollY + buttonRect.height / 2,
    };
  }, []);

  const drawLines = useCallback((startPositions) => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('line').remove(); // Clear existing lines before redrawing

    startPositions.forEach(startPos => {
      hiddenLayer.forEach((_, hiddenIndex) => {
        if (hiddenLayerActiveStatus[hiddenIndex]) {
          const endPos = calculateEndPos(hiddenIndex);
          const line = svg.append('line')
            .attr('x1', startPos.x)
            .attr('y1', startPos.y)
            .attr('x2', startPos.x)
            .attr('y2', startPos.y)
            .attr('stroke', 'black')
            .attr('stroke-width', 1);
            // .attr('stroke-dasharray', '2,2');

          line.transition()
            .duration(500)
            .attr('x2', endPos.x)
            .attr('y2', endPos.y);

          if (rightActiveIndex !== null) {
            const outputEndPos = calculateOutputEndPos(rightActiveIndex);
            const outputLine = svg.append('line')
              .attr('x1', endPos.x)
              .attr('y1', endPos.y)
              .attr('x2', endPos.x)
              .attr('y2', endPos.y)
              .attr('stroke', 'black')
              .attr('stroke-width', 1);
              // .attr('stroke-dasharray', '2,2');

            outputLine.transition()
              .duration(500)
              .attr('x2', outputEndPos.x)
              .attr('y2', outputEndPos.y);
          }
        }
      });
    });
  }, [calculateEndPos, calculateOutputEndPos, hiddenLayerActiveStatus, rightActiveIndex]);

  useEffect(() => {
    const fetchModelSummary = async () => {
      const response = await fetch('http://localhost:5000/model-summary', {
        method: 'GET',
      });

      if (response.ok) {
        const summary = await response.json();
        setEmbeddingLayer(Array.from({ length: summary.embedding_layer.output_features }, (_, index) => index + 1));
        setHiddenLayer(Array.from({ length: summary.hidden_layer.output_features }, (_, index) => index + 1));
        setOutputLayer(Array.from({ length: summary.output_layer.output_features }, (_, index) => index + 1));
      }
    };

    fetchModelSummary();
  }, []);

  useEffect(() => {
    if (positions.length > 0) {
      setTimeout(() => {
        drawLines(positions);
      }, 0);
    }
  }, [positions, drawLines]);

  useEffect(() => {
    const calculateAndSetPositions = () => {
      const newPositions = calculatePositions(embeddingLayerActiveStatus, 'button-embedding');
      setPositions(newPositions);
    };

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', calculateAndSetPositions);
    } else {
      calculateAndSetPositions();
    }

    return () => {
      document.removeEventListener('DOMContentLoaded', calculateAndSetPositions);
    };
  }, [embeddingLayerActiveStatus, calculatePositions]);

  return (
    <div className="App">
      <Header />
      <svg ref={svgRef} className="absolute top-0 left-0 w-full h-full" style={{ zIndex: 1, pointerEvents: 'none' }}></svg>
      <div className="flex relative">
        <div className="flex">
          <div className="flex flex-col">
            <div className="flex flex-col items-center h-full embedding-layer-margins ">
              <div className="text-center font-300 font-bold my-4">{"INPUT LAYER:"}</div>
              {embeddingLayer.map((_, index) => (
                <div key={index} className="my-2">
                  <Button
                    id={`button-embedding-${index}`}
                    isActive={embeddingLayerActiveStatus[index]}
                    onClick={() => handleEmbeddingLayerButtonClick(index)}
                  />
                </div>
              ))}
            </div>
          </div>
          <div className="flex flex-col items-center h-full previous-state">
            
            {extensionActiveIndex.map((isActive, index) => (
              <div key={index} className="my-2">
                <Button2
                  id={`button-extension-${index}`}
                  isActive={isActive}
                  onClick={() => handleExtensionButtonClick(index)}
                />
              </div>

            ))}
            {/* <div className="text-center font-300 font-bold my-4">{"PREVIOUS STATE:"}</div> */}
          </div>
        </div>
        <div className="flex flex-col items-center justify-center h-full right-space ">
          <div className="text-center font-300 font-bold my-4">{"HIDDEN LAYER:"}</div>
          {hiddenLayer.map((_, index) => (
            <div key={index} className="my-2">
              <Button
                id={`button-hidden-${index}`}
                isActive={hiddenLayerActiveStatus[index]}
                onClick={() => handleHiddenLayerButtonClick(index)}
              />
            </div>
          ))}
        </div>
        <div className="flex flex-col items-center h-full right-space">
          <div className="text-center font-300 font-bold my-4">{"OUTPUT LAYER:"}</div>
          {outputLayer.map((_, index) => (
            <div key={index} className="my-2">
              <Button
                id={`button-output-${index}`}
                isActive={rightActiveIndex === index}
                onClick={() => handleRightButtonClick(index)}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;


