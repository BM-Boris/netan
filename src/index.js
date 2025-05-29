// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { createTheme, responsiveFontSizes, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Create a theme instance.
let theme = createTheme({
  palette: {
    primary: {
      main: '#b88983', // Custom primary color (e.g., dark blue)
    },
    secondary: {
      main: '#597D94', // Custom secondary color (e.g., pink)
    },
    success: {
      main: '#83b889', // Custom success color (e.g., green)
    },
    // You can also customize error, warning, info, etc.
    error: {
      main: '#732d2d',
    },
    warning: {
      main: '#ff9800',
    },
    info: {
      main: '#83b2b8',
    },
  },
  typography: {
    fontFamily: '"Secular One", sans-serif',
    },
});
theme = responsiveFontSizes(theme);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <App />
  </ThemeProvider>
);
