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
                  <p>
                Provide one or more CSV files containing your feature and sample data:
              </p>
              <ul>
                <li><strong>Mass Spectrometry</strong>: Untargeted m/z, retention time, and intensities.</li>
                <li><strong>Transcriptomics / Genomics / Targeted MS</strong>: Identifier column + sample columns.</li>
                <li><strong>Meta Data</strong>: Sample annotation table (first column = sample IDs).</li>
                <li><strong>Other</strong>: Any numeric feature matrix.</li>
              </ul>
              
              <table className="table">
                <thead>
                  <tr>
                    <th style={{ paddingRight: '20px' }}>m/z</th>
                    <th style={{ paddingRight: '40px' }}>rt</th>
                    <th style={{ paddingRight: '20px' }}>sample1</th>
                    <th style={{ paddingRight: '20px' }}>sample2</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>85.02</td>
                    <td>62.3</td>
                    <td>2723031</td>
                    <td>2182606.4</td>
                  </tr>
                  <tr>
                    <td>86.06</td>
                    <td>293.2</td>
                    <td>3011843.5</td>
                    <td>2931308.9</td>
                  </tr>
                </tbody>
              </table>
              <p>
                <strong>Samples table</strong> must contain sample IDs as the first column and class labels for other columns:
              </p>
              <table className="table">
                <thead>
                  <tr>
                    <th style={{ paddingRight: '20px' }}>sample id</th>
                    <th style={{ paddingRight: '20px' }}>dose</th>
                    <th>time</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td >sample1</td>
                    <td >5mg</td>
                    <td>1</td>
                  </tr>
                  <tr>
                    <td >sample2</td>
                    <td >10mg</td>
                    <td>4</td>
                  </tr>
                </tbody>
              </table>
              <p>
                Sample columns are automatically aligned across all files; any sample not present in every file will be removed.
              </p>
              <p>
                Use +/– to add or remove entries. Enable <em>"Sync Settings"</em> to copy your preprocessing and filter choices to every non-meta file automatically.
              </p>
            
              <p>
                <strong>*</strong> Feature indexes in results match their position (starting from zero) in the original data.
              </p>
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
