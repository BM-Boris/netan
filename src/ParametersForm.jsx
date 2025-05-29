import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardContent,
  Typography,
  TextField,
  MenuItem,
  Box,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  IconButton,
  Collapse,
  Divider,
  Switch,
  Grow
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

// ────────────────────────────────────────────────────────────────────────────────
// CONSTANT MAPS – edge‑threshold defaults + lower bounds ("starting point")
// ────────────────────────────────────────────────────────────────────────────────
const EDGE_THRESHOLD_DEFAULTS = {
  spearman: 0.75,
  clr: 2.5,
  rf: 0.025,
  glasso:0.1,
};

const EDGE_THRESHOLD_MIN = {
  spearman: 0.3,
  clr: 1.5,
  rf: 0.001,
  glasso: 0,
};

// ────────────────────────────────────────────────────────────────────────────────
// NEW → stats‑threshold defaults per chosen method
// (keys follow backend codes via `normalizationMap`)
// ────────────────────────────────────────────────────────────────────────────────
const STATS_THRESHOLD_DEFAULTS = {
  none: 0.05,
  ttest: 0.05,
  oneway_anova: 0.05,
  pls_da: 2,
  fold_change: 1.5,
  sf_lg: 0.05,          // Logistic Regression
  sf_lr: 0.05,          // Linear Regression
  rf_class: 0.1,        // RF Classifier feature‑importance cutoff
  rf_regress: 0.1       // RF Regressor feature‑importance cutoff
};

// === Preprocessing params ===
const PREPROCESSING_PARAMS = [
  { name: 'threshold', label: 'Missing Filter', type: 'number', default: 1, step: 0.1 },
  { name: 'normalization', label: 'Normalization', type: 'select', options: ['none','quantile','total intensity','median','mean'], default: 'none' },
  { name: 'logTransformation', label: 'Transformation', type: 'select', options: ['none','log2'], default: 'none' },
  { name: 'scaling', label: 'Scaling', type: 'select', options: ['none','unit variance','pareto','minmax'], default: 'none' }
];

// === Filter params ===
const FILTER_PARAMS = [
  { name: 'method', label: 'Method', type: 'select',
    options: [
      'none', 
      't-test',
      'one-way ANOVA',
      'PLS-DA',
      'Log Fold Change',
      'Logistic Regression',
      'Linear Regression',
      'Random Forest Classifier',
      'Random Forest Regressor'
    ],
    default: 'none'
  },
  { name: 'meta', label: 'Meta Column', type: 'text', default: '' },
  { name: 'Threshold', label: 'Stats Threshold', type: 'number', step: 0.05, default: 0.05 },
];

// === Network params ===
const NETWORK_PARAMS = [
  {
    name: 'networkMethod', 
    label: 'Method', 
    type: 'select', 
    options: ['spearman','CLR (MI z-score)','Random-Forest Similarity','Graphical Lasso'], 
    default: 'spearman',
  },
  {
    name: 'nodeMode', 
    label: 'Node Mode', 
    type: 'select', 
    options: ['features','samples'], 
    default: 'samples'
  },
  {
    name: 'layerMode', 
    label: 'Layer Mode', 
    type: 'select', 
    options: ['stack','multilayer'], 
    default: 'stack'
  },
  {
    name: 'combineSamples', 
    label: 'Aggregation', 
    type: 'select', 
    options: ['mean','median','max'], 
    default: 'mean'
  },
  { name: 'edgeThreshold', label: 'Edge Threshold', type: 'number', step: 0.01, default: 0.75 },
  {
    name: 'layout', 
    label: 'Layout', 
    type: 'select', 
    options: ['force-directed','circular','kamada_kawai','random'], 
    default: 'force-directed'
  },
  {
    name: 'weights', 
    label: 'Edge Weights', 
    type: 'select', 
    options: ['on','off'], 
    default: 'on'
  }
];

/* --- ДОБАВЛЕННЫЕ метод‑специфичные поля ------------------------- */
const EXTRA_NET_PARAMS = [
  { name: 'n_neighbors',  label: 'n Neighbors', type: 'number',
    default: 2,  min: 1,   max: 20,   step: 1,  methods: ['clr'] },

  { name: 'n_estimators', label: 'n Estimators', type: 'number',
    default: 80, min: 10,  max: 500,  step: 10, methods: ['rf'] },

  { name: 'max_depth',    label: 'Max Depth',    type: 'number',
    default: 0,  min: 0,   max: 50,   step: 1,  methods: ['rf'] },

  { name: 'glassoAlpha',  label: 'Alpha',        type: 'number',
    default: 0.05, min: 0.0001, max: 1, step: 0.01, methods: ['glasso'] },

  { name: 'glassoMaxIter',label: 'Max Iter',     type: 'number',
    default: 200, min: 50, max: 1000, step: 10,   methods: ['glasso'] }
];
const NETWORK_PARAMS_CORE = NETWORK_PARAMS;
// Mapping for filter methods and normalization options
const normalizationMap = {
  'Logistic Regression': 'sf_lg',
  'Linear Regression': 'sf_lr',
  'Random Forest Classifier': 'rf_class',
  'Random Forest Regressor': 'rf_regress',
  'Log Fold Change': 'fold_change',
  'PLS-DA': 'pls_da',
  't-test': 'ttest',
  'one-way ANOVA': 'oneway_anova',
  none: 'none',
  quantile: 'q',
  'total intensity': 't',
  'unit variance': 'uv',
  spearman: 'spearman',
  'CLR (MI z-score)': 'clr',
  'Random-Forest Similarity': 'rf',
  'Graphical Lasso': 'glasso',
};

const labelWidth = 130;
const fixedWidth = 150;
const commonHeight = 40;
const sectionTitleStyle = { fontWeight: 'bold', mb: 1 };

// Build a param object for a single file
function buildSingleFileParams() {
  const dataObj = {};
  PREPROCESSING_PARAMS.forEach((p) => {
    dataObj[p.name] = p.default;
  });
  FILTER_PARAMS.forEach((p) => {
    if (p.name === 'Threshold') {
      dataObj[p.name] = STATS_THRESHOLD_DEFAULTS[normalizationMap[p.default] || p.default] || p.default;
    } else {
      dataObj[p.name] = p.default;
    }
  });
  return {
    data: dataObj,
    preprocessingOn: true,
    filterOn: true
  };
}

const ParametersForm = ({ onChangeParams, syncAll = true, files = [] }) => {
  // Identify non-meta files
  const nonMetaFiles = files.filter(
    (f) => f.type && f.type.toLowerCase() !== 'meta' && f.file
  );

  // Build a single network param object
  const initNetwork = {};
  [...NETWORK_PARAMS_CORE, ...EXTRA_NET_PARAMS].forEach(p => {
    initNetwork[p.name] = p.default;
  });
  initNetwork.edgeThreshold =
    EDGE_THRESHOLD_DEFAULTS[initNetwork.networkMethod];

  const [networkParams, setNetworkParams] = useState(initNetwork);

  // In sync mode, use a single parameter object; in non-sync, use an array (always at least one).
  const [preFilterParams, setPreFilterParams] = useState(() => {
    if (syncAll) return buildSingleFileParams();
    return nonMetaFiles.length > 0
      ? nonMetaFiles.map(() => buildSingleFileParams())
      : [buildSingleFileParams()];
  });

  // For collapsible sections: a single bool if syncAll, else an array.
  const [openPreprocessing, setOpenPreprocessing] = useState(() => (syncAll ? true : nonMetaFiles.length > 0 ? nonMetaFiles.map(() => true) : [true]));
  const [openFilter, setOpenFilter] = useState(() => (syncAll ? true : nonMetaFiles.length > 0 ? nonMetaFiles.map(() => true) : [true]));

  // Refs to track previous non-meta count and sync mode
  const prevNonMetaCountRef = useRef(nonMetaFiles.length);
  const prevSyncRef = useRef(syncAll);

  // Add state to control network parameters card visibility
  const [showNetwork, setShowNetwork] = useState(true);

  // Hide network parameters while switching sync mode, then show after a delay
  useEffect(() => {
    setShowNetwork(false);
    const timer = setTimeout(() => {
      setShowNetwork(true);
    }, 25);
    return () => clearTimeout(timer);
  }, [syncAll]);

  // Adjust parameter objects when sync mode or file count changes
  useEffect(() => {
    const oldCount = prevNonMetaCountRef.current;
    const newCount = nonMetaFiles.length;
    const oldSync = prevSyncRef.current;
    const newSync = syncAll;

    if (!oldSync && newSync) {
      // Switching from non-sync to sync: unify to the first parameter object.
      if (Array.isArray(preFilterParams) && preFilterParams.length > 0) {
        setPreFilterParams(preFilterParams[0]);
        if (Array.isArray(openPreprocessing) && openPreprocessing.length > 0) setOpenPreprocessing(openPreprocessing[0]);
        if (Array.isArray(openFilter) && openFilter.length > 0) setOpenFilter(openFilter[0]);
      } else {
        setPreFilterParams(buildSingleFileParams());
        setOpenPreprocessing(true);
        setOpenFilter(true);
      }
    } else if (oldSync && !newSync) {
      // Switching from sync to non-sync: replicate single object.
      if (!Array.isArray(preFilterParams)) {
        const singleObj = preFilterParams;
        const newArr = new Array(newCount || 1).fill(null).map(() =>
          JSON.parse(JSON.stringify(singleObj))
        );
        setPreFilterParams(newArr);
        const preVal = typeof openPreprocessing === 'boolean' ? openPreprocessing : true;
        const filtVal = typeof openFilter === 'boolean' ? openFilter : true;
        setOpenPreprocessing(new Array(newCount || 1).fill(preVal));
        setOpenFilter(new Array(newCount || 1).fill(filtVal));
      }
    }

    if (!newSync && Array.isArray(preFilterParams)) {
      if (newCount > oldCount) {
        const diff = newCount - oldCount;
        setPreFilterParams((prev) => {
          const out = [...prev];
          for (let i = 0; i < diff; i++) {
            out.push(buildSingleFileParams());
          }
          return out;
        });
        setOpenPreprocessing((prev) => {
          const out = Array.isArray(prev) ? [...prev] : [];
          for (let i = 0; i < diff; i++) {
            out.push(true);
          }
          return out;
        });
        setOpenFilter((prev) => {
          const out = Array.isArray(prev) ? [...prev] : [];
          for (let i = 0; i < diff; i++) {
            out.push(true);
          }
          return out;
        });
      } else if (newCount < oldCount) {
        setPreFilterParams((prev) => prev.slice(0, newCount || 1));
        setOpenPreprocessing((prev) => (Array.isArray(prev) ? prev.slice(0, newCount || 1) : prev));
        setOpenFilter((prev) => (Array.isArray(prev) ? prev.slice(0, newCount || 1) : prev));
      }
    }

    prevNonMetaCountRef.current = newCount;
    prevSyncRef.current = newSync;
  }, [syncAll, nonMetaFiles, preFilterParams, openPreprocessing, openFilter]);

  // Local dialog state for instructions
  const [dialogOpen, setDialogOpen] = useState(false);
  const handleDialogOpen = () => setDialogOpen(true);
  const handleDialogClose = () => setDialogOpen(false);

  // Helper to strip sections that are toggled off before sending to parent
  const stripSections = (obj) => {
    const clonedData = { ...obj.data };
    if (!obj.preprocessingOn) {
      PREPROCESSING_PARAMS.forEach((p) => delete clonedData[p.name]);
    }
    if (!obj.filterOn) {
      FILTER_PARAMS.forEach((p) => delete clonedData[p.name]);
    }
    return {
      data: clonedData,
      preprocessingOn: obj.preprocessingOn,
      filterOn: obj.filterOn
    };
  };

  // Finalize and send parameters to parent
  const sendToParent = (updatedNetwork, updatedPreFilter) => {
    let finalPreFilter;
    if (syncAll) {
      finalPreFilter = stripSections(updatedPreFilter);
    } else {
      finalPreFilter = updatedPreFilter.map(stripSections);
    }
    onChangeParams?.({
      networkParams: { ...updatedNetwork },
      preFilterParams: finalPreFilter
    });
  };

  // Network param changes (handles dynamic edge‑threshold defaults/mins)
  const handleNetworkChange = (e) => {
    const { name, value } = e.target;
    setNetworkParams((prev) => {
      let updated = { ...prev, [name]: value };

      // When the network method changes → update edge‑threshold default
      if (name === 'networkMethod') {
        updated.edgeThreshold = EDGE_THRESHOLD_DEFAULTS[value];
      }

      sendToParent(updated, preFilterParams);
      return updated;
    });
  };

  // In sync mode: update single preFilter data (incl. stats‑threshold defaults)
  const handleSinglePreFilterDataChange = (e) => {
    const { name, value } = e.target;
    setPreFilterParams((prev) => {
      const copy = { ...prev, data: { ...prev.data, [name]: value } };

      // If stats method changed → reset threshold default
      if (name === 'method') {
        const def = STATS_THRESHOLD_DEFAULTS[value] ?? 1;
        copy.data['Threshold'] = def;
      }

      sendToParent(networkParams, copy);
      return copy;
    });
  };

  const handleSinglePreprocessingToggle = (e) => {
    const { checked } = e.target;
    setPreFilterParams((prev) => {
      const copy = { ...prev, preprocessingOn: checked };
      sendToParent(networkParams, copy);
      return copy;
    });
  };

  const handleSingleFilterToggle = (e) => {
    const { checked } = e.target;
    setPreFilterParams((prev) => {
      const copy = { ...prev, filterOn: checked };
      sendToParent(networkParams, copy);
      return copy;
    });
  };

  // In non-sync mode: update specific card data
  const handleMultiPreFilterDataChange = (idx, e) => {
    const { name, value } = e.target;
    setPreFilterParams((prev) => {
      if (!Array.isArray(prev) || !prev[idx]) return prev;
      const newArr = [...prev];
      newArr[idx] = {
        ...newArr[idx],
        data: { ...newArr[idx].data, [name]: value }
      };

      if (name === 'method') {
        const def = STATS_THRESHOLD_DEFAULTS[value] ?? 1;
        newArr[idx].data['Threshold'] = def;
      }

      sendToParent(networkParams, newArr);
      return newArr;
    });
  };

  const handleMultiPreprocessingToggle = (idx, e) => {
    const { checked } = e.target;
    setPreFilterParams((prev) => {
      if (!Array.isArray(prev) || !prev[idx]) return prev;
      const newArr = [...prev];
      newArr[idx] = { ...newArr[idx], preprocessingOn: checked };
      sendToParent(networkParams, newArr);
      return newArr;
    });
  };

  const handleMultiFilterToggle = (idx, e) => {
    const { checked } = e.target;
    setPreFilterParams((prev) => {
      if (!Array.isArray(prev) || !prev[idx]) return prev;
      const newArr = [...prev];
      newArr[idx] = { ...newArr[idx], filterOn: checked };
      sendToParent(networkParams, newArr);
      return newArr;
    });
  };

  // Collapsibles
  const togglePreprocessing = (idx = 0) => {
    if (syncAll) {
      setOpenPreprocessing((prev) => (typeof prev === 'boolean' ? !prev : true));
    } else {
      if (!Array.isArray(openPreprocessing)) return;
      const arr = [...openPreprocessing];
      arr[idx] = !arr[idx];
      setOpenPreprocessing(arr);
    }
  };

  const toggleFilter = (idx = 0) => {
    if (syncAll) {
      setOpenFilter((prev) => (typeof prev === 'boolean' ? !prev : true));
    } else {
      if (!Array.isArray(openFilter)) return;
      const arr = [...openFilter];
      arr[idx] = !arr[idx];
      setOpenFilter(arr);
    }
  };

  // Render a block of parameter fields
  const renderParamFields = (paramDefs, values, onChangeFn) => {
    const isFilterSection = paramDefs === FILTER_PARAMS;
    const selectedMethod = values['method'];
    const disableFilterExtras = isFilterSection && selectedMethod === 'none';
    // Текущий код метода сети: 'spearman' / 'clr' / 'rf' / 'glasso'
    const currentNetLabel = values['networkMethod'] || networkParams.networkMethod;
    const methodCode      = normalizationMap[currentNetLabel] || currentNetLabel;

    return (
      <Grid container spacing={2} sx={{ mt: 1 }}>
        {paramDefs.map((pDef) => {
          const byMethod = pDef.methods && !pDef.methods.includes(methodCode);
          const isCombine = pDef.name === 'combineSamples';
          const aggregationDisabled = isCombine && (values['nodeMode'] !== 'samples' || values['layerMode'] !== 'multilayer');

          const isEdgeThreshold = pDef.name === 'edgeThreshold';
          const currentMin = isEdgeThreshold ? EDGE_THRESHOLD_MIN[values['networkMethod'] || networkParams.networkMethod] : undefined;

          const isMetaOrThreshold = pDef.name === 'meta' || pDef.name === 'Threshold';
          const shouldDisable =  byMethod || aggregationDisabled || (isMetaOrThreshold && disableFilterExtras);

          return (
            <Grid item xs={12} sm={6} md={4} key={pDef.name}>
              <Box display="flex" alignItems="center" gap={1}>
                <Typography sx={{ width: labelWidth }}>
                  {pDef.label}:
                </Typography>
                {pDef.type === 'select' ? (
                  // 1) Select-поле без изменений
                  <TextField
                    select
                    name={pDef.name}
                    value={values[pDef.name]}
                    onChange={onChangeFn}
                    size="small"
                    disabled={shouldDisable}
                    sx={{
                      height: commonHeight,
                      minWidth: fixedWidth,
                      width: fixedWidth
                    }}
                  >
                    {pDef.options.map((opt) => (
                      <MenuItem key={opt} value={normalizationMap[opt] || opt}>
                        {opt.charAt(0).toUpperCase() + opt.slice(1)}
                      </MenuItem>
                    ))}
                  </TextField>

                ) : pDef.type === 'number' ? (
                  // 2) Числовое поле с ограничителями
                  <TextField
                    name={pDef.name}
                    type="number"
                    value={values[pDef.name]}
                    onChange={onChangeFn}
                    onBlur={(e) => {
                      let v = parseFloat(e.target.value);
                      if (Number.isNaN(v)) return;
                      const minVal = pDef.min !== undefined
                        ? pDef.min 
                        : (pDef.name === 'edgeThreshold' ? currentMin : undefined);
                      const maxVal = pDef.max;
                      if (minVal !== undefined && v < minVal) v = minVal;
                      if (maxVal !== undefined && v > maxVal) v = maxVal;
                      if (v !== parseFloat(values[pDef.name])) {
                        e.target.value = v;
                        onChangeFn({ target: { name: pDef.name, value: v } });
                      }
                    }}
                    inputProps={{
                      step:      pDef.step ?? 'any',
                      min:       pDef.min !== undefined ? pDef.min : currentMin,
                      max:       pDef.max,
                      inputMode: 'numeric',
                      pattern:   '[0-9]*'
                    }}
                    onKeyDown={(e) => {
                      const allowed = ['Backspace','Tab','ArrowLeft','ArrowRight','Delete','Home','End','.'];
                      if (!/[0-9]/.test(e.key) && e.key !== '.' && !allowed.includes(e.key)) {
                        e.preventDefault();
                      }
                    }}
                    size="small"
                    disabled={shouldDisable}
                    sx={{
                      height: commonHeight,
                      minWidth: fixedWidth,
                      width: fixedWidth
                    }}
                  />

                ) : (
                  // 3) Простое текстовое поле (для Meta Column)
                  <TextField
                    name={pDef.name}
                    type="text"
                    value={values[pDef.name]}
                    onChange={onChangeFn}
                    size="small"
                    disabled={shouldDisable}
                    sx={{
                      height: commonHeight,
                      minWidth: fixedWidth,
                      width: fixedWidth
                    }}
                  />

                )}
              </Box>
            </Grid>
          );
        })}
      </Grid>
    );
  };

  // Render a single file's parameter card (wrapped in Grow for a smooth appearance)
  const renderOneFileCard = (title, paramObj, idx) => {
    if (!paramObj) return null;
    const isPreOpen = syncAll
      ? (typeof openPreprocessing === 'boolean' ? openPreprocessing : true)
      : (Array.isArray(openPreprocessing) && openPreprocessing[idx] !== undefined ? openPreprocessing[idx] : true);
    const isFilterOpen = syncAll
      ? (typeof openFilter === 'boolean' ? openFilter : true)
      : (Array.isArray(openFilter) && openFilter[idx] !== undefined ? openFilter[idx] : true);

    const togglePre = () => togglePreprocessing(idx);
    const toggleFilt = () => toggleFilter(idx);
    const handlePreToggle = (e) => {
      if (syncAll) handleSinglePreprocessingToggle(e);
      else handleMultiPreprocessingToggle(idx, e);
    };
    const handleFilterToggle = (e) => {
      if (syncAll) handleSingleFilterToggle(e);
      else handleMultiFilterToggle(idx, e);
    };
    const handleParamChange = (e) => {
      if (syncAll) handleSinglePreFilterDataChange(e);
      else handleMultiPreFilterDataChange(idx, e);
    };

    return (
      <Grow in={true} timeout={500} key={title}>
        <Card
          sx={{
            borderRadius: 2,
            boxShadow: 2,
            p: 2,
            backgroundColor: '#f3efeb',
            width: '100%',
            maxWidth: 1000,
            mb: 3
          }}
        >
          <Box display="flex" alignItems="center" mb={2}>
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
              {title}
            </Typography>
            <IconButton onClick={handleDialogOpen} color="primary" sx={{ ml: 1 }}>
              <HelpOutlineIcon />
            </IconButton>
          </Box>

          {/* Preprocessing Section */}
          <Box display="flex" alignItems="center" sx={{ cursor: 'pointer' }} onClick={togglePre}>
            <Typography variant="subtitle1" sx={sectionTitleStyle}>
              Preprocessing
            </Typography>
            <Box ml={3.5} mb={1.4} onClick={(e) => e.stopPropagation()}>
              <Switch
                checked={paramObj.preprocessingOn}
                onChange={handlePreToggle}
              />
            </Box>
            <Box flexGrow={1} />
            {isPreOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </Box>
          <Collapse in={isPreOpen} timeout="auto" unmountOnExit>
            {paramObj.preprocessingOn && renderParamFields(PREPROCESSING_PARAMS, paramObj.data, handleParamChange)}
          </Collapse>

          <Divider sx={{ my: 2 }} />

          {/* Filter Section */}
          <Box display="flex" alignItems="center" sx={{ cursor: 'pointer' }} onClick={toggleFilt}>
            <Typography variant="subtitle1" sx={sectionTitleStyle}>
              Filter Selection
            </Typography>
            <Box ml={3.5} mb={1.4} onClick={(e) => e.stopPropagation()}>
              <Switch
                checked={paramObj.filterOn}
                onChange={handleFilterToggle}
              />
            </Box>
            <Box flexGrow={1} />
            {isFilterOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </Box>
          <Collapse in={isFilterOpen} timeout="auto" unmountOnExit>
            {paramObj.filterOn && renderParamFields(FILTER_PARAMS, paramObj.data, handleParamChange)}
          </Collapse>
        </Card>
      </Grow>
    );
  };

  // Render network parameters card (wrapped in Grow)
  const renderNetworkCard = () => (
    <Grow in={true} timeout={500}>
      <Card
        sx={{
          borderRadius: 2,
          boxShadow: 2,
          p: 2,
          backgroundColor: '#f3efeb',
          width: '100%',
          maxWidth: 1000,
          mb: 3
        }}
      >
        <Box display="flex" alignItems="center" mb={2}>
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
            Network Parameters
          </Typography>
          <IconButton onClick={handleDialogOpen} color="primary" sx={{ ml: 1 }}>
            <HelpOutlineIcon />
          </IconButton>
        </Box>
        {renderParamFields(
            [...NETWORK_PARAMS_CORE, ...EXTRA_NET_PARAMS],  // ← было только NETWORK_PARAMS_CORE
            networkParams,
            handleNetworkChange
        )}
      </Card>
    </Grow>
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 4 }}>
      {syncAll ? (
        renderOneFileCard('Parameters (All Non-Meta Files)', preFilterParams, 0)
      ) : (
        Array.isArray(preFilterParams) &&
        preFilterParams.map((paramObj, idx) => {
          const fileName = (nonMetaFiles[idx] && nonMetaFiles[idx].file?.name) || `File ${idx + 1}`;
          return renderOneFileCard(`Parameters - ${fileName}`, paramObj, idx);
        })
      )}
      {/* Conditionally render network parameters */}
      {showNetwork && renderNetworkCard()}

      <Dialog open={dialogOpen} onClose={handleDialogClose}>
        <DialogTitle>Parameter Instructions</DialogTitle>
        <DialogContent>
          <DialogContentText>
            <ul>
              <li>
                <strong>Preprocessing:</strong> threshold, normalization,
                transformation, scaling
              </li>
              <li>
                <strong>Filter Selection:</strong> method, meta column, Threshold
                (capital T)
              </li>
              <li>
                <strong>Network:</strong> method, edge threshold, layout
              </li>
            </ul>
            <p>
              This version preserves your values even when files are added or
              removed.
            </p>
          </DialogContentText>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default ParametersForm;
