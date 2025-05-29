/* NetworkPlot.jsx – single‑layer selector + uid sanitising
   =================================================================
   • Починка MUI‑warning «out-of-range value» + fallback MenuItem ''
=================================================================== */

import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import {
  Card, Typography, Box, FormControl, InputLabel, Select, MenuItem,
  Switch, FormControlLabel, IconButton, Dialog, Slider
} from '@mui/material';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';

/* ───────── helpers ───────── */
const NUM_UNIQUE_THRESHOLD = 6;
const isNum   = v => v !== null && v !== '' && !Number.isNaN(+v);
const colType = (col, data) => {
  if (!col) return 'none';
  const vals = data.map(d => d[col]).filter(v => v !== undefined && v !== null);
  if (!vals.length) return 'none';
  return vals.every(isNum) && new Set(vals.map(Number)).size >= NUM_UNIQUE_THRESHOLD
    ? 'continuous' : 'categorical';
};
const palette  = ['red','blue','green','orange','purple','brown','pink','gray','teal','gold',
                  'cyan','magenta','lime','olive','maroon','navy','silver','violet','indigo',
                  'coral','khaki','turquoise','tan'];
const symbols  = ['circle','square','diamond','triangle-up','triangle-down','cross','x','star'];
const safeId   = s => String(s).replace(/[^A-Za-z0-9_-]/g, '_');

/* ───────── control panel ───────── */
const Panel = ({
  columns, colorBy, setColorBy,
  shapeBy, setShapeBy,
  selLayer, setSelLayer, layerNames,
  hideIso, setHideIso,
  minW, maxW, stepW, range, setRange
}) => (
  <Box display="flex" alignItems="center" gap={3} mb={2} flexWrap="wrap">
    {/* Color */}
    <FormControl size="small">
      <InputLabel>Color by</InputLabel>
      <Select value={columns.includes(colorBy) ? colorBy : ''}
              onChange={e => setColorBy(e.target.value)}
              label="Color by" sx={{ width: 150 }}>
        <MenuItem value="">(None)</MenuItem>
        {columns.map(c => <MenuItem key={c} value={c}>{c}</MenuItem>)}
      </Select>
    </FormControl>

    {/* Shape */}
    <FormControl size="small">
      <InputLabel>Shape by</InputLabel>
      <Select value={columns.includes(shapeBy) ? shapeBy : ''}
              onChange={e => setShapeBy(e.target.value)}
              label="Shape by" sx={{ width: 150 }}>
        <MenuItem value="">(None)</MenuItem>
        {columns.map(c => <MenuItem key={c} value={c}>{c}</MenuItem>)}
      </Select>
    </FormControl>

    {/* Layer */}
    <FormControl size="small" disabled={layerNames.length === 0}>
      <InputLabel>Layer</InputLabel>
      <Select
        /* гарантируем, что value существует в списке;
          если нет — берём первый доступный слой */
        value={layerNames.includes(selLayer) ? selLayer : (layerNames[0] || '')}
        onChange={e => setSelLayer(e.target.value)}
        label="Layer"
        sx={{ width: 150 }}
      >
        {layerNames.map(l => (
          <MenuItem key={l} value={l}>{l}</MenuItem>
        ))}
      </Select>
    </FormControl>

    {/* Hide isolated */}
    <FormControlLabel
      control={<Switch checked={hideIso} onChange={e => setHideIso(e.target.checked)}
                       color="secondary" sx={{ mb: 0.4 }} />}
      label="Hide isolated"
    />

    {/* Weight slider */}
    <Box sx={{ width: 220, px: 1 }}>
      <Typography variant="body2" sx={{ mb: 0.3 }}>Edge weight range</Typography>
      <Slider value={range} onChange={(_, v) => setRange(v)} valueLabelDisplay="auto"
              min={minW} max={maxW} step={stepW} marks color="secondary" sx={{ mt: -0.8 }} />
      <Box display="flex" justifyContent="space-between" sx={{ mt: -1 }}>
        <Typography variant="caption">{minW.toFixed(3)}</Typography>
        <Typography variant="caption">{maxW.toFixed(3)}</Typography>
      </Box>
    </Box>
  </Box>
);

/* ───────── main component ───────── */
const NetworkPlot = ({ nodes, edges }) => {
  /* dynamic column list */
  const cols = Object.keys(nodes[0] || {}).filter(
    k => !['id','x','y','compound','display_id'].includes(k)
  );

  /* state */
  const [colorBy, setColorBy] = useState('');
  const [shapeBy, setShapeBy] = useState('');
  const [hideIso, setHideIso] = useState(false);
  const [full, setFull] = useState(false);
  const toggleFull = () => setFull(v => !v);

  /* reset color/shape when columns change */
  useEffect(() => {
    if (!cols.includes(colorBy)) setColorBy('');
    if (!cols.includes(shapeBy)) setShapeBy('');
  }, [cols]);                                                // eslint-disable-line

  /* layer names */
  const layerNames = useMemo(
    () => [...new Set(edges.flatMap(e => e.layers || [e.layer]))].sort(),
    [edges]
  );
  const [selLayer, setSelLayer] = useState('');
  useEffect(() => {
    // pick first layer automatically or reset to ''
    setSelLayer(p => (layerNames.includes(p) ? p : (layerNames[0] || '')));
  }, [layerNames]);

  /* hidden legend items */
  const [hidden, setHidden] = useState(new Set());
  useEffect(() => setHidden(new Set()), [colorBy, shapeBy]);

  /* weight slider bounds */
  const wArr  = useMemo(() => edges.map(e => e.weight), [edges]);
  const minW  = useMemo(() => wArr.length ? +Math.min(...wArr).toFixed(3) : 0, [wArr]);
  const maxW  = useMemo(() => wArr.length ? +Math.max(...wArr).toFixed(3) : 1, [wArr]);
  const stepW = useMemo(() => +((maxW - minW) / 19).toFixed(3) || 0.001, [minW, maxW]);
  const [range, setRange] = useState([minW, maxW]);
  useEffect(() => setRange([minW, maxW]), [minW, maxW]);

  /* zoom memory */
  const axisRef = useRef(null);

  /* grouping key */
  const gKey = useCallback(n => {
    const c = colorBy ? n[colorBy] : 'all';
    const s = shapeBy ? n[shapeBy] : 'all';
    if (colorBy && shapeBy) return `${c}||${s}`;
    if (colorBy)            return `${c}`;
    if (shapeBy)            return `${s}`;
    return 'Nodes';
  }, [colorBy, shapeBy]);

  /* maps */
  const maps = useMemo(() => {
    const keys  = [...new Set(nodes.map(gKey))].sort();
    const cVals = [...new Set(nodes.map(n => colorBy ? n[colorBy] : 'all'))].sort();
    const sVals = [...new Set(nodes.map(n => shapeBy ? n[shapeBy] : 'all'))].sort();
    const cMap  = {}, sMap = {};
    cVals.forEach((v,i) => { cMap[v] = palette[i % palette.length]; });
    sVals.forEach((v,i) => { sMap[v] = symbols[i % symbols.length]; });
    return { keys, cMap, sMap, cType: colType(colorBy, nodes) };
  }, [nodes, colorBy, shapeBy, gKey]);

  /* traces */
  const data = useMemo(() => {
    /* filter edges by layer + weight */
    const eFilt = edges.filter(e => {
      const list = e.layers || [e.layer];
      return (!selLayer || list.includes(selLayer)) &&
             e.weight >= range[0] && e.weight <= range[1];
    });

    /* nodes/edges visibility */
    let vNodes = nodes.filter(n => !hidden.has(gKey(n)));
    const idSet = new Set(vNodes.map(n => n.id));
    let vEdges = eFilt.filter(({source,target}) => idSet.has(source)&&idSet.has(target));
    if (hideIso) {
      const conn = new Set();
      vEdges.forEach(e => { conn.add(e.source); conn.add(e.target); });
      vNodes = vNodes.filter(n => conn.has(n.id));
      const id2 = new Set(vNodes.map(n => n.id));
      vEdges = vEdges.filter(({source,target}) => id2.has(source)&&id2.has(target));
    }

    /* edges trace */
    const pos = Object.fromEntries(vNodes.map(n => [n.id, {x:n.x, y:n.y}]));
    const ex=[], ey=[];
    vEdges.forEach(({source,target})=>{
      const s=pos[source], t=pos[target]; if(!s||!t) return;
      ex.push(s.x,t.x,null); ey.push(s.y,t.y,null);
    });
    const traces=[{
      uid:'edges', x:ex, y:ey, mode:'lines', hoverinfo:'none',
      showlegend:false, line:{color:'#888',width:1}
    }];

    /* nodes */
    if (maps.cType==='continuous') {
      const byShape={};
      vNodes.forEach(n=>{
        const sv=shapeBy ? n[shapeBy] : 'all';
        if(!byShape[sv]) byShape[sv]={x:[],y:[],c:[],t:[]};
        byShape[sv].x.push(n.x); byShape[sv].y.push(n.y);
        byShape[sv].c.push(+n[colorBy]);
        byShape[sv].t.push(`ID: ${n.display_id}<br>${colorBy}: ${n[colorBy]}`);
      });
      Object.entries(byShape).forEach(([sv,g],i)=>{
        const safe=safeId(sv);
        traces.push({
          uid:`nodes_${safe}`, x:g.x, y:g.y, mode:'markers',
          hoverinfo:'text', text:g.t, showlegend:false,
          marker:{
            color:g.c, colorscale:'Viridis', showscale:i===0,
            colorbar:i===0?{title:colorBy}:undefined,
            symbol:shapeBy ? maps.sMap[sv] : 'circle',
            size:10, line:{width:1,color:'#333'}
          }
        });
      });
      return traces;
    }

    /* categorical */
    const groups={};
    vNodes.forEach(n=>{
      const k=gKey(n);
      if(!groups[k]) groups[k]={x:[],y:[],t:[],cVal:colorBy?n[colorBy]:'all',
                                sVal:shapeBy?n[shapeBy]:'all'};
      groups[k].x.push(n.x); groups[k].y.push(n.y);
      let t=`ID: ${n.display_id}` +
        (n.compound ? `<br>Compound: ${n.compound}` : '');
      if(colorBy) t+=`<br>${colorBy}: ${groups[k].cVal}`;
      if(shapeBy) t+=`<br>${shapeBy}: ${groups[k].sVal}`;
      groups[k].t.push(t);
    });
    maps.keys.forEach(k=>{
      const g=groups[k]||{x:[null],y:[null],t:[],cVal:null,sVal:null};
      const safe=safeId(k);
      traces.push({
        uid:`nodes_${safe}`, x:g.x, y:g.y, mode:'markers',
        name:k, legendgroup:k,
        hoverinfo:'text', text:g.t,
        visible:hidden.has(k)?'legendonly':true,
        marker:{
          color:g.cVal!=null?maps.cMap[g.cVal]:'#000',
          symbol:g.sVal!=null?maps.sMap[g.sVal]:'circle',
          size:10, line:{width:1,color:'#333'}
        }
      });
    });
    return traces;
  }, [nodes, edges, selLayer, range, hidden, hideIso, colorBy, shapeBy, maps, gKey]);

  /* legend interaction */
  const onLegendClick = useCallback(ev=>{
    const g=ev?.data?.[ev.curveNumber]?.name; if(!g) return false;
    setHidden(p=>{const n=new Set(p); n.has(g)?n.delete(g):n.add(g); return n;});
    return false;
  },[]);
  const onLegendDouble = useCallback(()=>{ setHidden(new Set()); return false; },[]);

  /* relayout – axis memory */
  const onRelayout = useCallback(ev=>{
    if('xaxis.range[0]' in ev){
      axisRef.current={
        x:[ev['xaxis.range[0]'],ev['xaxis.range[1]']],
        y:[ev['yaxis.range[0]'],ev['yaxis.range[1]']]
      };
    } else if('xaxis.autorange' in ev || 'yaxis.autorange' in ev){
      axisRef.current=null;
    }
  },[]);

  const legendTop = shapeBy && maps.cType==='continuous';
  const legendCfg = legendTop
    ? {orientation:'h',x:0.5,y:1.05,xanchor:'center',yanchor:'bottom',
       itemwidth:30,itemsizing:'trace',tracegroupgap:12}
    : {orientation:'v',x:1.02,y:1,xanchor:'left',tracegroupgap:8};

  const makeLayout = full => ({
    title:'Network Plot', hovermode:'closest', showlegend:true, legend:legendCfg,
    margin:full ? {l:20,r:20,t:40,b:20} :
                  {l:20,r:60,t:legendTop?70:40,b:20},
    xaxis:{visible:false,
           ...(axisRef.current?{range:axisRef.current.x,autorange:false}:{})},
    yaxis:{visible:false,
           ...(axisRef.current?{range:axisRef.current.y,autorange:false}:{})},
    uirevision:'network'
  });

  const PlotBox = ({full}) => (
    <Plot
      data={data}
      layout={makeLayout(full)}
      style={full
        ? {width:'100%',height:'calc(100vh - 200px)'}
        : {width:'100%',height:'600px'}}
      config={{responsive:true}}
      onLegendClick={onLegendClick}
      onLegendDoubleClick={onLegendDouble}
      onRelayout={onRelayout}
    />
  );

  /* ───────── UI ───────── */
  return (
    <>
      {/* NORMAL CARD */}
      <Box sx={{display:'flex',flexDirection:'column',alignItems:'center',mt:4}}>
        <Card sx={{borderRadius:2,boxShadow:2,p:2,backgroundColor:'#D7DFE3',
                   width:'100%',maxWidth:1000}}>
          <Box sx={{display:'flex',alignItems:'center',
                    justifyContent:'space-between',mb:2.5}}>
            <Typography variant="h5" sx={{fontWeight:'bold'}}>Network Plot</Typography>
            <IconButton onClick={toggleFull} size="small">
              <FullscreenIcon sx={{fontSize:30}}/>
            </IconButton>
          </Box>

          <Panel
            columns={cols}
            colorBy={colorBy}   setColorBy={setColorBy}
            shapeBy={shapeBy}   setShapeBy={setShapeBy}
            selLayer={selLayer} setSelLayer={setSelLayer} layerNames={layerNames}
            hideIso={hideIso}   setHideIso={setHideIso}
            minW={minW} maxW={maxW} stepW={stepW}
            range={range} setRange={setRange}
          />

          <PlotBox full={false}/>
        </Card>
      </Box>

      {/* FULLSCREEN DIALOG */}
      <Dialog fullScreen open={full} onClose={toggleFull}
              PaperProps={{sx:{backgroundColor:'#D7DFE3'}}}>
        <Box sx={{display:'flex',alignItems:'center',
                  justifyContent:'space-between',p:2,pt:3,mb:0.5}}>
          <Typography variant="h4" sx={{fontWeight:'bold'}}>Network Plot</Typography>
          <IconButton onClick={toggleFull}>
            <FullscreenExitIcon sx={{fontSize:38}}/>
          </IconButton>
        </Box>

        <Box sx={{px:2}}>
          <Panel
            columns={cols}
            colorBy={colorBy}   setColorBy={setColorBy}
            shapeBy={shapeBy}   setShapeBy={setShapeBy}
            selLayer={selLayer} setSelLayer={setSelLayer} layerNames={layerNames}
            hideIso={hideIso}   setHideIso={setHideIso}
            minW={minW} maxW={maxW} stepW={stepW}
            range={range} setRange={setRange}
          />

          <PlotBox full={true}/>
        </Box>
      </Dialog>
    </>
  );
};

export default NetworkPlot;
