// NetworkBuilder.jsx  — React front‑end for Rodin «Build‑network» API
// Реализует реальный детерминированный прогресс‑бар через task_id + polling
// ===========================================================================

import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Box,
  Button,
  Typography,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Card,
  Grow
} from '@mui/material';
import axios from 'axios';

import DataUploadWithType from './DataUploadWithType';
import ParametersForm from './ParametersForm';
import NetworkPlot from './NetworkPlot';

const API = 'http://10.66.12.172:80/api/build-network/';

const NetworkBuilder = () => {
  // ───────── Local state ───────────────────────────────────────────────────
  const [filesData, setFilesData] = useState([]);   // [{file, type}, …]
  const [syncAll, setSyncAll] = useState(true);
  const [parameters, setParameters] = useState({}); // Form values

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);      // 0–100
  const [taskId, setTaskId] = useState(null);       // uuid from server

  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [buildStats, setBuildStats] = useState(null);

  const [dataTable, setDataTable] = useState([]);

  const [errorMessage, setErrorMessage] = useState(null);
  const [errorDialogOpen, setErrorDialogOpen] = useState(false);

  const pollRef = useRef(null);                     // interval id

  // ───────── Handlers (file / params) ──────────────────────────────────────
  const handleFilesChange      = (info) => setFilesData(info);
  const handleSyncChange       = (v)   => setSyncAll(v);
  const handleParametersChange = (p)   => setParameters(p);

  const canBuild =
    filesData.length > 0 && filesData.every(f => f.file && f.type);

  // ───────── Polling helpers ───────────────────────────────────────────────
  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  

  const pollProgress = (id) => {
    pollRef.current = setInterval(async () => {
      try {
        const { data } = await axios.get(`${API}?task_id=${id}`);
        setProgress(data.progress ?? 0);
  
        if (data.status === 'done') {
          setNodes(data.result.nodes || []);
          setEdges(data.result.edges || []);
          if (data.result.dataTable) {
            setDataTable(data.result.dataTable);
          }
          if (data.result.stats) setBuildStats(data.result.stats);
  
          setLoading(false);
          stopPolling();
        } else if (data.status === 'error') {
          throw new Error(data.error || 'Unknown server error');
        } else if (data.status === 'cancelled' || data.status === 'cancelling') {
          setLoading(false);
          stopPolling();
        }
      } catch (err) {
        setErrorMessage(err.message);
        setErrorDialogOpen(true);
        setLoading(false);
        stopPolling();
      }
    }, 2500); // poll every 2.5 s
  };

  // ───────── Build button click ────────────────────────────────────────────
  const handleBuildNetwork = async () => {
    stopPolling();
    setLoading(true);
    setProgress(0);
    setNodes([]);
    setEdges([]);
    setBuildStats(null);

    const formData = new FormData();
    filesData.forEach(f => {
      formData.append('data_files', f.file);
      formData.append('file_types', f.type);
    });
    formData.append('parameters',
      JSON.stringify({ syncAll, paramData: parameters }));

    try {
      const res = await axios.post(API, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const id = res.data.task_id;
      setTaskId(id);
      pollProgress(id);
    } catch (err) {
      setErrorMessage(
        err.response?.data?.error ? err.response.data.error : err.message
      );
      setErrorDialogOpen(true);
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    try {
      await axios.patch(`${API}?task_id=${taskId}&action=cancel`);
    } catch (err) { /* тихо игнорируем */ }
    stopPolling();
    setLoading(false);
    setProgress(0);
    setTaskId(null);
    setNodes([]); setEdges([]); setBuildStats(null);
  };

  // clear polling on component unmount
  useEffect(() => stopPolling, []);

  useEffect(() => {
    const abortOnUnload = () => {
      if (!taskId) return;
      const url = `${API}?task_id=${taskId}&action=cancel`;
      // пустой Blob – GET‑запрос; можно отправить FormData → будет POST
      navigator.sendBeacon(url);
    };
    window.addEventListener('beforeunload', abortOnUnload);
    return () => window.removeEventListener('beforeunload', abortOnUnload);
  }, [taskId]);

  // ───────── Build‑stats pretty printer (same as ваш оригинал) ─────────────
  const getStatsText = () => {
    if (!buildStats) return '';
    const lines = [];

    /* per‑file */
    if (Array.isArray(buildStats.fileStats)) {
      buildStats.fileStats.forEach((s, i) => {
        const orig    = s.features_original        ?? 0;
        const prep    = s.features_after_preproc   ?? orig;
        const remPrep = s.features_removed_preproc ?? orig - prep;

        const beforeFil = s.features_before_filter    ?? prep;
        const afterFil  = s.features_after_filter     ?? beforeFil;
        const remFil    = s.features_removed_filter   ?? beforeFil - afterFil;

        const shape = s.X_shape ? `(${s.X_shape.join('×')})` : '(?)';

        lines.push(
          `File #${i + 1}: ` +
          `original=${orig}, ` +
          `after preproc=${prep}(‑${remPrep}), ` +
          `after filter=${afterFil}(‑${remFil}), ` +
          `X.shape=${shape}`
        );
      });
    }

    /* network global */
    const net = buildStats.networkStats || {};
    lines.push('', 'Network Stats:');
    lines.push(` - Total nodes: ${net.numNodes ?? '?'}`);
    lines.push(` - Edges: ${net.numEdges ?? '?'}`);
    lines.push(` - Nodes w/ edges: ${net.numNodesWithEdges ?? '?'}`);
    lines.push(` - Density: ${(net.density ?? 0).toFixed(4)}`);
    lines.push(` - Components: ${net.numComponents ?? '?'}`);
    lines.push(` - Communities: ${net.numCommunities ?? '?'}`);

    /* per‑layer */
    const layerDict = {}; // {layer:{nodes,edges,density}}
    Object.entries(net).forEach(([k, v]) => {
      let m, lname;
      if ((m = k.match(/^nodes_(.+)/))) {
        lname = m[1];
        layerDict[lname] = { ...(layerDict[lname] || {}), nodes: v };
      } else if ((m = k.match(/^edges_(.+)/))) {
        lname = m[1];
        layerDict[lname] = { ...(layerDict[lname] || {}), edges: v };
      } else if ((m = k.match(/^density_(.+)/))) {
        lname = m[1];
        layerDict[lname] = {
          ...(layerDict[lname] || {}),
          density: typeof v === 'number' ? v.toFixed(4) : v
        };
      }
    });

    if (Object.keys(layerDict).length) {
      lines.push('', 'Per‑layer Stats:');
      Object.entries(layerDict).forEach(([l, s]) => {
        const label = l.replace(/_/g, ' ');
        lines.push(
          ` • ${label}: nodes=${s.nodes ?? '?'}, edges=${s.edges ?? '?'}, ` +
          `density=${s.density ?? '?'}`
        );
      });
    }

    return lines.join('\n');
  };

  // ───────── Download CSV (edges) ──────────────────────────────────────────
  const handleDownloadNetwork = () => {
    if (!edges.length) return;

    // header
    const lines = ['source,target,weight,layer,layers'];
    // rows
    edges.forEach(e => {
      const src   = e.source;
      const tgt   = e.target;
      const w     = e.weight ?? 1;
      const layer = e.layer ?? '';
      const lays  = Array.isArray(e.layers) ? e.layers.join('|') : '';
      lines.push([src, tgt, w, layer, lays].join(','));
    });

    const csv = lines.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'network_netan.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const handleDownloadData = () => {
    if (!dataTable.length) return;
  
    // 1) Собираем объединённый набор всех ключей из всех строк
    const headerSet = new Set();
    dataTable.forEach(row => {
      Object.keys(row).forEach(key => headerSet.add(key));
    });
    const headers = Array.from(headerSet);
  
    // 2) Строим CSV-строки
    const lines = [];
    // Заголовок
    lines.push(headers.join(','));
    // Данные
    dataTable.forEach(row => {
      const line = headers
        .map(h => {
          const v = row[h];
          // если undefined или null — оставляем пустым
          return v == null ? '' : String(v);
        })
        .join(',');
      lines.push(line);
    });
  
    // 3) Сохраняем как файл
    const csv = lines.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'rodin_data.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  
  

  // ───────── JSX ───────────────────────────────────────────────────────────
  return (
    <Container sx={{ mt: 1 }}>
      {/* Upload + Sync */}
      <DataUploadWithType
        onFilesChange={handleFilesChange}
        onSyncChange={handleSyncChange}
      />

      {/* Parameters form */}
      <ParametersForm
        syncAll={syncAll}
        files={filesData}
        onChangeParams={handleParametersChange}
      />

      {/* Build button */}
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
        <Button
          variant="contained"
          color="secondary"
          onClick={handleBuildNetwork}
          disabled={!canBuild || loading}
          sx={{ textTransform: 'none', width: 200, fontSize: '1.2rem' }}
        >
          {loading ? 'Building…' : 'Build'}
        </Button>
      </Box>

      {/* Linear progress (determinate) */}
      {loading && (
        <Box sx={{ width: '100%', mt: 3.5 }}>
          <LinearProgress
            color="secondary"
            variant="buffer"
            value={progress}
            valueBuffer={progress+5}
          />
          <Typography align="center" sx={{ mt: 0.5 }}>
            {progress.toFixed(0)} %
          </Typography>
        </Box>
      )}

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
          <Button
            variant="outlined"
            color="error"
            onClick={handleCancel}
            sx={{ textTransform: 'none', width: 140 }}
          >
            Cancel
          </Button>
        </Box>
      )}



      {/* Build Stats card */}
      <Grow in={Boolean(buildStats)} timeout={800}>
        <Box sx={{ mt: 4 }}>
          {buildStats && (
            <Card
              sx={{
                p: 2,
                backgroundColor: '#D7DFE3',
                maxWidth: 1000,
                m: '0 auto'
              }}
            >
              <Typography
                variant="h5"
                sx={{ fontWeight: 'bold', mb: 2 }}
                gutterBottom
              >
                Build Stats
              </Typography>
              <Typography
                component="pre"
                sx={{
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'monospace',
                  fontSize: '0.95rem',
                  lineHeight: 1.5
                }}
              >
                {getStatsText()}
              </Typography>
            </Card>
          )}
        </Box>
      </Grow>

      {/* Network plot + Download button */}
      <Grow in={nodes.length && edges.length} timeout={800}>
        <Box sx={{ mt: 4 }}>
          <NetworkPlot nodes={nodes} edges={edges} />
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <Button
              color="secondary"
              variant="outlined"
              onClick={handleDownloadNetwork}
              sx={{ textTransform: 'none' }}
            >
              Download Network
            </Button>
            </Box>
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <Button
              variant="outlined"
              onClick={handleDownloadData}
              disabled={!dataTable.length}
              sx={{ textTransform: 'none' }}
            >
            Download Data
          </Button>
          </Box>
        </Box>
      </Grow>

      {/* Error dialog */}
      <Dialog
        open={errorDialogOpen}
        onClose={() => setErrorDialogOpen(false)}
      >
        <DialogTitle>Error</DialogTitle>
        <DialogContent>{errorMessage}</DialogContent>
        <DialogActions>
          <Button onClick={() => setErrorDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default NetworkBuilder;
