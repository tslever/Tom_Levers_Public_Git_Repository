/**
 * A Shell organizes one or more Organizers into a payload to be delivered to a client.
 * A Shell provides the rules and mechanisms for how Organizers are made available to a user.
 * A Shell provides no functionality other than that required to implement those rules and mechanisms.
 * A Shell manages how one or more Organizers are made visible to a user (e.g., simultaneously or by user selection).
 * A Shell is a React component.
 * A Shell is the top-level React component of a payload.
 * The artifact of building a Shell is a single HTML file to be delivered to a client when it makes a request to an endpoint (e.g., http://localhost:3000).
 * A Shell with two or more Organizers has a background color and specifies characteristics for each HTML division that bounds an Organizer.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
