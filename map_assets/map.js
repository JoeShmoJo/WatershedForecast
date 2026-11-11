'use strict';
try {
  const data = window.BASIN_MAP;
  const text = value => document.createTextNode(value);
  const format = value => new Date(value).toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'});
  document.getElementById('map-title').textContent = `${data.name} · ${data.hours}h precipitation`;
  document.getElementById('map-period').textContent = `${format(data.start)} → ${format(data.end)}`;
  const map = L.map('map',{scrollWheelZoom:false});
  map.createPane('precipitation').style.zIndex=350;
  const tiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',{
    maxZoom:16,
    attribution:'Tiles &copy; Esri. Sources: Esri, HERE, Garmin, Intermap, increment P Corp., GEBCO, USGS, FAO, NPS, NRCAN, GeoBase, IGN, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), OpenStreetMap contributors, and the GIS User Community'
  }).addTo(map);
  map.attributionControl.addAttribution('Forecast: NOAA NBM');
  let tileError;
  tiles.on('tileerror',()=>{
    if(tileError)return;
    tileError=L.control({position:'bottomleft'});
    tileError.onAdd=()=>{const el=L.DomUtil.create('div','tile-error');el.textContent='Background map unavailable; precipitation and basin boundaries still shown.';return el};
    tileError.addTo(map);
  });
  const overlay=L.imageOverlay('precip_overlay.png',data.bounds,{pane:'precipitation',opacity:1}).addTo(map);
  const outlines=L.geoJSON(data.features,{
    style:feature=>({color:feature.properties.parent?'#087ca5':'#213e4c',weight:feature.properties.parent?2.8:1.5,fillOpacity:0}),
    onEachFeature:(feature,layer)=>{
      const label=feature.properties;
      const content=document.createElement('div');
      const name=document.createElement('strong');name.textContent=label.name;
      content.append(name,document.createElement('br'),text(`${label.total_inches.toFixed(2)} in basin-average precipitation`));
      layer.bindPopup(content);
      if(!label.parent){
        const tooltip=document.createElement('span');tooltip.textContent=`${label.name} · ${label.total_inches.toFixed(2)} in`;
        layer.bindTooltip(tooltip,{permanent:true,direction:'center',className:'subbasin-label'});
      }
    }
  }).addTo(map);
  function fit(){map.fitBounds(data.bounds,{padding:[26,26]})} fit();
  const controls=L.control({position:'topright'});
  controls.onAdd=()=>{
    const el=L.DomUtil.create('div','map-controls');
    const button=document.createElement('button');button.type='button';button.textContent='Fit basin';button.onclick=fit;
    const label=document.createElement('label');const box=document.createElement('input');box.type='checkbox';box.checked=true;
    box.onchange=()=>{if(box.checked)overlay.addTo(map);else overlay.remove()};label.append(box,text('Precipitation'));
    el.append(button,label);L.DomEvent.disableClickPropagation(el);L.DomEvent.disableScrollPropagation(el);return el;
  };controls.addTo(map);
  [...data.levels].reverse().forEach((level,j)=>{
    const i=data.levels.length-1-j,row=document.createElement('div'),swatch=document.createElement('span');
    row.className='legend-row';swatch.style.background=data.colors[i];row.append(swatch,text(`${level}${i===data.levels.length-1?'+':''}`));document.getElementById('legend').append(row);
  });
  document.getElementById('dry').hidden=data.maximum_inches>=.01;
  // Exposed solely for inspection/testing; display uses precomputed NBM totals.
  window.basinMap={map,overlay,outlines};
} catch(error) {
  document.getElementById('map-error').hidden=false;
  console.error(error);
}
