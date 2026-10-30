"""Inline renderer for portable neighborhood reports (no external assets)."""

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Genomic neighborhoods · meetneighbors</title>
<style>
:root{color-scheme:light;--ink:#18263d;--muted:#64748b;--line:#dfe6ed;--accent:#5268cb}
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:14px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1600px;margin:36px auto;padding:0 28px 50px}.eyebrow{font-size:11px;letter-spacing:.15em;color:var(--muted);font-weight:700;text-transform:uppercase}
h1{font-size:30px;letter-spacing:-.035em;margin:8px 0}p{margin:6px 0}.intro{color:var(--muted);max-width:850px}
.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:16px 20px;background:#fff;border:1px solid var(--line);border-radius:12px 12px 0 0;margin-top:26px}
.count{font-weight:650;margin-right:auto}button{border:1px solid #cbd5e1;border-radius:6px;padding:5px 12px;background:#fff;color:var(--ink);cursor:pointer;font:inherit}button:hover{background:#eef2ff}button:focus-visible,input:focus-visible{outline:3px solid #b9c4ff;outline-offset:2px}
input{accent-color:var(--accent);width:115px}.zoom-value{min-width:36px;font-variant-numeric:tabular-nums}
.scroller{overflow-x:auto;background:#fff;border:1px solid var(--line);border-top:0;border-radius:0 0 12px 12px}
.axis{display:block;background:#f8fafc;border-bottom:1px solid var(--line)}.row{padding-top:16px;border-bottom:1px solid var(--line)}.row:last-child{border:0}
.row-head{padding:0 28px}.name{margin:0;font-size:13px;font-weight:650;overflow-wrap:anywhere}.meta{font-size:11px;color:var(--muted);margin-top:3px}.gene-track{display:block}
.gene{cursor:pointer;outline:none;transition:opacity .12s}.gene.active{stroke:#14263c;stroke-width:2.5;filter:drop-shadow(0 2px 2px #1e293b33)}.gene.dimmed{opacity:.18}.gene:focus-visible{stroke:#111827;stroke-width:3}
.legend{margin-top:20px;background:white;border:1px solid var(--line);border-radius:10px;padding:14px 18px}.legend summary{cursor:pointer;font-weight:650}.legend-items{display:flex;flex-wrap:wrap;gap:9px 22px;margin-top:12px}.legend-item{display:flex;align-items:center;gap:7px;max-width:100%;font-size:12px;overflow-wrap:anywhere}.swatch{flex:none;width:18px;height:9px;border-radius:2px}.anchor-swatch{background:#e2e8f0;border:2px solid #243451}
.note{font-size:12px;color:var(--muted);margin:14px 2px}.tooltip{position:fixed;z-index:10;pointer-events:none;max-width:min(440px,calc(100vw - 24px));background:#17243a;color:white;padding:15px 17px;border-radius:9px;box-shadow:0 8px 28px #0f172a40;font-size:12px;overflow-wrap:anywhere}.tooltip[hidden]{display:none}.tooltip strong{font-size:14px;display:block;margin-bottom:9px}.tooltip .detail{color:#d4dcec;white-space:pre-wrap}
@media(max-width:600px){main{padding:0 12px;margin-top:22px}h1{font-size:25px}.toolbar{padding:12px;gap:8px}}
</style>
</head>
<body><main>
<div class="eyebrow">meetneighbors / genomic context</div>
<h1>Neighborhoods, side by side</h1>
<p class="intro">Aligned at the center gene, oriented to the right. Matching colors mark protein clusters shared across neighborhoods. Hover or focus a gene to inspect its original annotation.</p>
<div class="toolbar"><span class="count" id="count"></span><label for="zoom">Zoom</label><button id="zoom-out" aria-label="Zoom out">−</button><input type="range" id="zoom" min="1" max="8" step="0.25" value="1"><button id="zoom-in" aria-label="Zoom in">+</button><span class="zoom-value" id="zoom-value">1×</span><button id="reset">Reset</button></div>
<div class="scroller" id="scroller" tabindex="0" aria-label="Genomic neighborhood tracks; scroll horizontally when zoomed"><div id="canvas"></div></div>
<details class="legend" open><summary id="legend-title">Protein clusters</summary><div id="legend-items" class="legend-items"></div></details>
<p class="note" id="status"></p><p class="note">Lengths and spacing are shown in nucleotides. Gray genes have no cluster shared across displayed neighborhoods. Dark outlines identify center genes; original strand and coordinates are retained in the tooltip.</p>
<noscript>This interactive report requires JavaScript enabled in your browser.</noscript>
</main><div id="tooltip" class="tooltip" role="tooltip" hidden></div>
<script id="neighborhood-data" type="application/json">__NEIGHBORHOOD_DATA__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('neighborhood-data').textContent);
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const palette = ['#e7b65f','#65b7a0','#8699d3','#d48fb0','#95b86a','#ca9876','#6badc4','#bba0d2','#cebd64','#80b9bd','#d89989','#99a8bd'];
const colors = new Map(data.shared_clusters.map((c,i) => [c, palette[i] || `hsl(${(i*137.508)%360} 48% 62%)`]));
const allGenes = data.tracks.flatMap(t => t.genes);
let minimum = Math.min(...allGenes.map(g=>g.left)), maximum = Math.max(...allGenes.map(g=>g.right));
const span = Math.max(maximum-minimum,1); minimum -= span*.04; maximum += span*.04;
let zoom=1, polygons=[];
function node(tag, attrs={}, text=null){const el=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!==null)el.textContent=text;return el;}
function element(tag, className, text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
function details(g,t){return `Protein: ${g.protein}\nCluster: ${g.cluster || 'Unclustered'}\nGenome: ${g.genome}\nContig: ${g.contig}\nCoordinates: ${g.start.toLocaleString()}–${g.end.toLocaleString()} (1-based, inclusive)\nStrand: ${g.strand}${t.flipped ? ' · display reversed' : ''}\nCenter gene: ${g.anchor ? 'Yes' : 'No'}\nNeighborhood: ${t.name}`;}
function moveTip(x,y){const tip=$('tooltip');const box=tip.getBoundingClientRect();tip.style.left=Math.max(12,Math.min(x+16,innerWidth-box.width-12))+'px';tip.style.top=Math.max(12,Math.min(y+16,innerHeight-box.height-12))+'px';}
function showTip(g,t,el,event){const tip=$('tooltip');tip.replaceChildren(element('strong','',g.product),element('div','detail',details(g,t)));tip.hidden=false;const box=el.getBoundingClientRect();moveTip(event.clientX ?? box.x,event.clientY ?? box.bottom);for(const item of polygons){const same=item.el===el || (g.cluster!==null && item.gene.cluster===g.cluster);item.el.classList.toggle('active',same);item.el.classList.toggle('dimmed',!same);}el.setAttribute('aria-describedby','tooltip');}
function hideTip(){ $('tooltip').hidden=true;for(const item of polygons){item.el.classList.remove('active','dimmed');item.el.removeAttribute('aria-describedby');}}
function draw(){hideTip();const canvas=$('canvas');canvas.replaceChildren();polygons=[];const width=Math.max(900,$('scroller').clientWidth)*zoom;canvas.style.width=width+'px';const x=v=>28+(v-minimum)/(maximum-minimum)*(width-56);
 const axis=node('svg',{width,height:54,class:'axis','aria-label':'Distance from center gene in nucleotides'});axis.append(node('text',{x:28,y:17,fill:'#64748b','font-size':10},'RELATIVE POSITION · NT'));
 const rough=(maximum-minimum)/Math.max(3,Math.floor(width/130));const power=10**Math.floor(Math.log10(rough));const step=[1,2,5,10].map(n=>n*power).find(n=>n>=rough);
 for(let tick=Math.ceil(minimum/step)*step;tick<=maximum;tick+=step){axis.append(node('line',{x1:x(tick),x2:x(tick),y1:35,y2:41,stroke:'#bdc8d6'}),node('text',{x:x(tick),y:32,'text-anchor':'middle',fill:'#64748b','font-size':11},Math.round(tick).toLocaleString()));}
 axis.append(node('line',{x1:x(0),x2:x(0),y1:37,y2:54,stroke:'#5268cb','stroke-width':2}));canvas.append(axis);
 for(const track of data.tracks){const row=element('section','row');const head=element('div','row-head');head.append(element('h2','name',track.name),element('div','meta',`${track.genome} · ${track.contig} · ${track.genes.length} genes${track.flipped?' · reversed to orient center →':''}`));row.append(head);const height=36+track.lanes*27;const svg=node('svg',{width,height,class:'gene-track',role:'group','aria-label':track.name});svg.append(node('line',{x1:x(0),x2:x(0),y1:0,y2:height,stroke:'#e7ecfc','stroke-width':10}));
  for(let lane=0;lane<track.lanes;lane++)svg.append(node('line',{x1:28,x2:width-28,y1:24+lane*27,y2:24+lane*27,stroke:'#eff2f6'}));
  for(const gene of track.genes){const left=x(gene.left),right=x(gene.right),y=24+gene.lane*27,tip=Math.min(9,(right-left)*.45),h=7;const points=gene.direction>0?`${left},${y-h} ${right-tip},${y-h} ${right},${y} ${right-tip},${y+h} ${left},${y+h}`:`${right},${y-h} ${left+tip},${y-h} ${left},${y} ${left+tip},${y+h} ${right},${y+h}`;
   const polygon=node('polygon',{points,fill:colors.get(gene.cluster)||'#dce1e8',stroke:gene.anchor?'#243451':'#fff','stroke-width':gene.anchor?2:1,class:'gene',tabindex:0,role:'img','aria-label':`${gene.protein}: ${gene.product}. ${details(gene,track)}`});polygon.dataset.protein=gene.protein;if(gene.cluster!==null)polygon.dataset.cluster=gene.cluster;polygon.addEventListener('pointerenter',e=>showTip(gene,track,polygon,e));polygon.addEventListener('pointermove',e=>moveTip(e.clientX,e.clientY));polygon.addEventListener('pointerleave',hideTip);polygon.addEventListener('focus',e=>showTip(gene,track,polygon,e));polygon.addEventListener('blur',hideTip);svg.append(polygon);polygons.push({el:polygon,gene});
   if(gene.anchor)svg.append(node('circle',{cx:(left+right)/2,cy:y-13,r:3,fill:'#5268cb','pointer-events':'none'}));
  }row.append(svg);canvas.append(row);
 }
}
$('count').textContent=`${data.tracks.length} neighborhoods · ${allGenes.length} gene occurrences`;
$('status').textContent=data.message+(data.recluster_settings?` Identity ≥ ${data.recluster_settings.seq_id}; coverage ≥ ${data.recluster_settings.cov}; coverage mode ${data.recluster_settings.cov_mode}.`:'');
$('legend-title').textContent=`Shared protein clusters (${data.shared_clusters.length})`;
function legend(color,label,anchor=false){const item=element('span','legend-item');const swatch=element('span','swatch'+(anchor?' anchor-swatch':''));swatch.style.backgroundColor=color;item.append(swatch,element('span','',label));$('legend-items').append(item);}
for(const [cluster,color] of colors)legend(color,cluster);
legend('#dce1e8','Not shared / unclustered');legend('#dce1e8','Center gene',true);
function setZoom(value){const oldWidth=$('canvas').clientWidth;const center=($('scroller').scrollLeft+$('scroller').clientWidth/2)/oldWidth;zoom=Math.min(8,Math.max(1,value));$('zoom').value=zoom;$('zoom-value').textContent=zoom+'×';draw();$('scroller').scrollLeft=center*$('canvas').clientWidth-$('scroller').clientWidth/2;}
$('zoom').addEventListener('input',e=>setZoom(Number(e.target.value)));
$('zoom-in').addEventListener('click',()=>setZoom(zoom+.5));$('zoom-out').addEventListener('click',()=>setZoom(zoom-.5));$('reset').addEventListener('click',()=>{setZoom(1);$('scroller').scrollLeft=0;});
document.addEventListener('keydown',e=>{if(e.key==='Escape')hideTip();});window.addEventListener('resize',draw);$('scroller').addEventListener('scroll',hideTip);draw();
</script></body></html>
'''
