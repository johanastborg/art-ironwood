'use client';

import { useState } from 'react';

interface Sphere {
  center: [number, number, number];
  radius: number;
  color: [number, number, number];
  reflectivity: number;
  roughness: number;
  transmission: number;
  ior: number;
}

export default function Home() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [lightPos, setLightPos] = useState<[number, number, number]>([5.0, 10.0, -5.0]);

  const [spheres, setSpheres] = useState<Sphere[]>([
    {
      center: [-1.2, 0.5, -3.0],
      radius: 0.5,
      color: [1.0, 0.0, 0.0],
      reflectivity: 0.5,
      roughness: 0.1,
      transmission: 0.0,
      ior: 1.0,
    },
    {
      center: [0.0, 0.5, -3.0],
      radius: 0.5,
      color: [0.0, 1.0, 0.0],
      reflectivity: 0.5,
      roughness: 0.4,
      transmission: 0.0,
      ior: 1.0,
    },
    {
      center: [1.2, 0.5, -3.0],
      radius: 0.5,
      color: [0.0, 0.0, 1.0],
      reflectivity: 0.5,
      roughness: 0.05,
      transmission: 0.0,
      ior: 1.0,
    },
    {
      center: [0.0, 1.0, -1.5],
      radius: 1.0,
      color: [1.0, 0.0, 1.0],
      reflectivity: 0.1,
      roughness: 0.05,
      transmission: 0.75,
      ior: 1.5,
    },
  ]);

  const handleSphereChange = (index: number, field: keyof Sphere, value: any) => {
    const newSpheres = [...spheres];
    newSpheres[index] = { ...newSpheres[index], [field]: value };
    setSpheres(newSpheres);
  };

  const handleColorChange = (index: number, channel: number, val: number) => {
    const newSpheres = [...spheres];
    const newColor = [...newSpheres[index].color] as [number, number, number];
    newColor[channel] = val;
    newSpheres[index] = { ...newSpheres[index], color: newColor };
    setSpheres(newSpheres);
  };

  const renderScene = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/render', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          spheres: spheres,
          light_pos: lightPos,
        }),
      });

      if (!response.ok) {
        console.error("Failed to fetch image");
        return;
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setImageUrl(url);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif', maxWidth: '1400px', margin: '0 auto' }}>
      <h1>Avantime Ray Tracer 🐀✨</h1>

      <div style={{ display: 'flex', gap: '20px', flexDirection: 'row' }}>
        <div style={{ flex: 1, minWidth: '400px', overflowY: 'auto', maxHeight: '90vh' }}>
          <h2>Controls</h2>

          <div style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px', borderRadius: '8px' }}>
            <h3>Light Position</h3>
            <div style={{ display: 'flex', gap: '10px' }}>
              <label>X: <input type="number" step="0.1" value={lightPos[0]} onChange={(e) => setLightPos([parseFloat(e.target.value), lightPos[1], lightPos[2]])} style={{width: '60px'}} /></label>
              <label>Y: <input type="number" step="0.1" value={lightPos[1]} onChange={(e) => setLightPos([lightPos[0], parseFloat(e.target.value), lightPos[2]])} style={{width: '60px'}} /></label>
              <label>Z: <input type="number" step="0.1" value={lightPos[2]} onChange={(e) => setLightPos([lightPos[0], lightPos[1], parseFloat(e.target.value)])} style={{width: '60px'}} /></label>
            </div>
          </div>

          {spheres.map((sphere, idx) => (
            <div key={idx} style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px', borderRadius: '8px' }}>
              <h3>Sphere {idx + 1}</h3>

              <div style={{ marginBottom: '10px' }}>
                <label>Color (RGB):</label><br/>
                <div style={{ display: 'flex', gap: '5px' }}>
                   <input type="number" step="0.1" min="0" max="1" value={sphere.color[0]} onChange={(e) => handleColorChange(idx, 0, parseFloat(e.target.value))} style={{width: '50px'}} />
                   <input type="number" step="0.1" min="0" max="1" value={sphere.color[1]} onChange={(e) => handleColorChange(idx, 1, parseFloat(e.target.value))} style={{width: '50px'}} />
                   <input type="number" step="0.1" min="0" max="1" value={sphere.color[2]} onChange={(e) => handleColorChange(idx, 2, parseFloat(e.target.value))} style={{width: '50px'}} />
                   <div style={{
                     width: '20px', height: '20px',
                     backgroundColor: `rgb(${sphere.color[0]*255}, ${sphere.color[1]*255}, ${sphere.color[2]*255})`,
                     border: '1px solid #000'
                   }}></div>
                </div>
              </div>

              <div style={{ marginBottom: '5px' }}>
                <label>Roughness (0-1):</label>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={sphere.roughness}
                  onChange={(e) => handleSphereChange(idx, 'roughness', parseFloat(e.target.value))}
                  style={{ marginLeft: '10px' }}
                />
                <span> {sphere.roughness}</span>
              </div>

              <div style={{ marginBottom: '5px' }}>
                <label>Reflectivity (0-1):</label>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={sphere.reflectivity}
                  onChange={(e) => handleSphereChange(idx, 'reflectivity', parseFloat(e.target.value))}
                  style={{ marginLeft: '10px' }}
                />
                <span> {sphere.reflectivity}</span>
              </div>

              <div style={{ marginBottom: '5px' }}>
                <label>Transmission (0-1):</label>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={sphere.transmission}
                  onChange={(e) => handleSphereChange(idx, 'transmission', parseFloat(e.target.value))}
                  style={{ marginLeft: '10px' }}
                />
                <span> {sphere.transmission}</span>
              </div>

              <div style={{ marginBottom: '5px' }}>
                <label>IOR:</label>
                <input
                  type="number" step="0.1"
                  value={sphere.ior}
                  onChange={(e) => handleSphereChange(idx, 'ior', parseFloat(e.target.value))}
                  style={{ width: '60px', marginLeft: '10px' }}
                />
              </div>

              <div style={{ marginBottom: '5px' }}>
                <label>Radius:</label>
                <input
                  type="number" step="0.1"
                  value={sphere.radius}
                  onChange={(e) => handleSphereChange(idx, 'radius', parseFloat(e.target.value))}
                  style={{ width: '60px', marginLeft: '10px' }}
                />
              </div>
              <div style={{ display: 'flex', gap: '5px' }}>
                <label>Pos (XYZ):</label>
                <input type="number" step="0.1" value={sphere.center[0]} onChange={(e) => handleSphereChange(idx, 'center', [parseFloat(e.target.value), sphere.center[1], sphere.center[2]])} style={{width: '50px'}} />
                <input type="number" step="0.1" value={sphere.center[1]} onChange={(e) => handleSphereChange(idx, 'center', [sphere.center[0], parseFloat(e.target.value), sphere.center[2]])} style={{width: '50px'}} />
                <input type="number" step="0.1" value={sphere.center[2]} onChange={(e) => handleSphereChange(idx, 'center', [sphere.center[0], sphere.center[1], parseFloat(e.target.value)])} style={{width: '50px'}} />
              </div>

            </div>
          ))}

          <button
            onClick={renderScene}
            disabled={loading}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              backgroundColor: loading ? '#ccc' : '#0070f3',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: loading ? 'not-allowed' : 'pointer',
              marginBottom: '20px'
            }}
          >
            {loading ? 'Rendering...' : 'Render Scene'}
          </button>
        </div>

        <div style={{ flex: 2, display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
          {imageUrl ? (
            <div style={{ border: '2px solid #333' }}>
              <img src={imageUrl} alt="Rendered Scene" style={{ maxWidth: '100%', height: 'auto' }} />
            </div>
          ) : (
             <div style={{ width: '800px', height: '600px', backgroundColor: '#eee', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
               <p>No render yet. Click "Render Scene".</p>
             </div>
          )}
        </div>
      </div>
    </div>
  );
}
