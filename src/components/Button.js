import React from 'react';

// Button component with forwardRef
const Button = React.forwardRef(({ isActive, onClick }, ref) => {
  return (
    <button
      ref={ref}
      className={`w-5 h-5 rounded ${isActive ? 'bg-orange-500' : 'bg-blue-500'}`}
      onClick={onClick}
    >
      {/* You can add additional button content or labels here */}
    </button>
  );
});

// Button2 component with forwardRef
const Button2 = React.forwardRef(({ isActive, onClick }, ref) => {
  return (
    <button
      ref={ref}
      className={`w-5 h-5 rounded ${isActive ? 'bg-orange-200' : 'bg-blue-200'}`}
      onClick={onClick}
    >
      {/* You can add additional button content or labels here */}
    </button>
  );
});

// Export both Button and Button2 components
export { Button, Button2 };