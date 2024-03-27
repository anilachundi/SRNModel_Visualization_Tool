import React from 'react';

// Button component
function Button({ isActive, onClick }) {
  return (
    <button
      className={`w-5 h-5 rounded ${isActive ? 'bg-orange-500' : 'bg-blue-500'}`}
      onClick={onClick}
    >
      {/* Add button content here */}
    </button>
  );
}

// Button2 component
function Button2({ isActive, onClick }) {
  return (
    <button
      className={`w-5 h-5 rounded ${isActive ? 'bg-orange-200' : 'bg-blue-200'}`}
      onClick={onClick}
    >
      {/* Add button content here */}
    </button>
  );
}

// Export both Button and Button2 components
export { Button, Button2 };