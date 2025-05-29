import React, { useState, useEffect } from 'react';
import { 
  Card, CardContent, Typography, Button, Box, TextField, MenuItem, IconButton, Paper, 
  Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions,
  Switch,Grow,
  FormControlLabel 
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const dataTypes = [
  { value: 'metabolomics', label: 'Mass Spectrometry Data' },
  { value: 'targeted_metabol', label: 'Targeted MS Data' },
  { value: 'transcriptomics', label: 'Transcriptomics' },
  { value: 'genomics', label: 'Genomics' },
  { value: 'meta', label: 'Meta Data' },
  { value: 'others', label: 'Other' },
];

function buildFileInfoArray(uploads) {
  return uploads.map((u) => ({ file: u.file, type: u.type, id: u.id }));
}

const DataUploadWithType = ({ onFilesChange, onSyncChange }) => {
  const [uploads, setUploads] = useState([{ id: Date.now(), file: null, type: '' }]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [fileNameWidth, setFileNameWidth] = useState(120);

  // Whether we are syncing Pre/Filter params for multiple non-meta files
  const [syncAll, setSyncAll] = useState(true);

  const commonHeight = 40; 
  const minFileNameWidth = 120;
  const maxFileNameLimit = 450;

  // Dynamically compute the "file name" display width
  useEffect(() => {
    const computed = uploads.reduce((acc, upload) => {
      if (upload.file) {
        // approximate 9px per character
        const width = upload.file.name.length * 9;
        return Math.max(acc, width);
      }
      return acc;
    }, minFileNameWidth);
    const finalWidth = Math.min(Math.max(computed, minFileNameWidth), maxFileNameLimit);
    setFileNameWidth(finalWidth);
  }, [uploads]);

  // Notify parent whenever syncAll changes
  useEffect(() => {
    onSyncChange?.(syncAll);
  }, [syncAll, onSyncChange]);

  // Update a single upload record
  const updateUpload = (id, key, value) => {
    setUploads((prev) => {
      const newUploads = prev.map((upload) =>
        upload.id === id ? { ...upload, [key]: value } : upload
      );
      onFilesChange?.(buildFileInfoArray(newUploads));
      return newUploads;
    });
  };

  const addUpload = () => {
    setUploads((prev) => {
      const newUploads = [...prev, { id: Date.now(), file: null, type: '' }];
      onFilesChange?.(buildFileInfoArray(newUploads));
      return newUploads;
    });
  };

  const removeUpload = (id) => {
    setUploads((prev) => {
      const newUploads = prev.filter((upload) => upload.id !== id);
      onFilesChange?.(buildFileInfoArray(newUploads));
      return newUploads;
    });
  };

  const handleDialogOpen = () => setDialogOpen(true);
  const handleDialogClose = () => setDialogOpen(false);

  // Toggle the Switch for “Sync / Not sync”
  const handleSyncToggle = (e) => {
    setSyncAll(e.target.checked);
  };

  // Count how many "non-meta" files
  const nonMetaFilesCount = uploads.filter(
    (u) => u.file && u.type && u.type.toLowerCase() !== 'meta'
  ).length;

  return (
    <Box sx={{ display: 'flex', justifyContent: 'center' }}>
      <Grow in={true} timeout={500}>
      <Card 
        sx={{ 
          mt: 4, 
          borderRadius: 2, 
          boxShadow: 2, 
          p: 2, 
          backgroundColor: '#f3efeb',
          width: '100%',
          maxWidth: 1000,
        }}
      >
       
          <Box display="flex" alignItems="center" mb={2}>
            <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
              Upload Data Files
            </Typography>
            <IconButton onClick={handleDialogOpen} color="primary" sx={{ ml: 1, mt: '-2px' }}>
              <HelpOutlineIcon />
            </IconButton>
          </Box>
          
          {uploads.map((upload) => (
            <Box
              key={upload.id}
              display="flex"
              flexDirection="row"
              flexWrap="wrap"
              alignItems="center"
              gap={3}
              mb={2}
            >
              {/* File upload button */}
              <Button 
                variant="contained" 
                component="label" 
                color="primary"
                startIcon={<CloudUploadIcon sx={{ mt: '-3px'}} />}
                sx={{ 
                  height: commonHeight,
                  textTransform: 'none',
                  alignItems: 'center',
                  minWidth: 150,
                }}
              >
                {upload.file ? 'Change File' : 'Select File'}
                <input
                  type="file"
                  accept=".csv,.txt"
                  hidden
                  onChange={(e) => updateUpload(upload.id, 'file', e.target.files[0])}
                />
              </Button>

              {/* Show selected filename */}
              {upload.file && (
                <Paper 
                  variant="outlined"
                  sx={{ 
                    p: 1,  
                    height: commonHeight, 
                    display: 'flex', 
                    alignItems: 'center',
                    width: fileNameWidth,
                    ...(fileNameWidth === maxFileNameLimit && {
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'normal'
                    })
                  }}
                >
                  <Typography variant="body2">
                    {upload.file.name}
                  </Typography>
                </Paper>
              )}

              {/* Data type dropdown */}
              <TextField
                select
                label="data type"
                value={upload.type}
                onChange={(e) => updateUpload(upload.id, 'type', e.target.value)}
                size="small"
                sx={{ 
                  minWidth: 180,
                  height: commonHeight,
                }}
              >
                {dataTypes.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </TextField>

              {/* Delete button (only if >1 files) */}
              {uploads.length > 1 && (
                <IconButton
                  onClick={() => removeUpload(upload.id)}
                  color="error"
                  sx={{ height: commonHeight }}
                >
                  <DeleteIcon sx={{ mb: '-2px'}} />
                </IconButton>
              )}
            </Box>
          ))}

          <Box mt={2} display="flex" alignItems="center" gap={3}>
            <Button
              variant="outlined"
              onClick={addUpload}
              startIcon={<AddIcon sx={{ mt: '-3px'}} />}
              sx={{ textTransform: 'none', height: commonHeight }}
            >
              Add more files
            </Button>

            {/* Disable switch if fewer than 2 non-meta files */}
            <FormControlLabel 
              label={syncAll ? 'Synced' : 'Not sync'}
              control={
                <Switch sx={{mb: 0.4 }}
                  checked={syncAll}
                  onChange={handleSyncToggle}
                  disabled={nonMetaFilesCount < 2}
                />
              }
              sx={{ userSelect: 'none', ml: -0.8 }}
            />
          </Box>
        

        {/* Instructions dialog */}
        <Dialog open={dialogOpen} onClose={handleDialogClose}>
          <DialogTitle>Data Upload Instructions</DialogTitle>
          <DialogContent>
            <DialogContentText>
              Please upload your data files according to the following guidelines:
              <ul>
                <li><strong>Metabolomics:</strong> CSV/TXT with metabolite measurements.</li>
                <li><strong>Transcriptomics:</strong> CSV/TXT with gene expression data.</li>
                <li><strong>Genomics:</strong> CSV/TXT with genetic variants or sequences.</li>
                <li><strong>Meta Data:</strong> Additional sample or experimental info.</li>
              </ul>
              You can add multiple files and specify a data type for each.
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleDialogClose} color="primary">Close</Button>
          </DialogActions>
        </Dialog>
      </Card>
      </Grow>
    </Box>
  );
};

export default DataUploadWithType;
