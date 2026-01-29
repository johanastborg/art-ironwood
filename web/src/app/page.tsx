'use client';

import { useState, useRef, useEffect, Suspense } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

interface Sphere {
  center: [number, number, number];
  radius: number;
  color: [number, number, number];
  reflectivity: number;
  roughness: number;
  transmission: number;
  ior: number;
}

function SceneContent({
  spheres,
  onInteractionStart,
  onInteractionEnd
}: {
  spheres: Sphere[],
  onInteractionStart: () => void,
  onInteractionEnd: (cam: THREE.Camera, target: THREE.Vector3) => void
}) {
  const controlsRef = useRef<any>(null);

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[5.0, 10.0, -5.0]} intensity={1.5} />

      <OrbitControls
        ref={controlsRef}
        onStart={onInteractionStart}
        onEnd={() => onInteractionEnd(controlsRef.current.object, controlsRef.current.target)}
        target={[0.0, 0.5, -3.0]}
      />

      <PerspectiveCamera makeDefault position={[0.0, 0.8, -0.5]} fov={75} />

      {/* Plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="gray" wireframe />
      </mesh>

      {/* Spheres */}
      {spheres.map((s, idx) => (
        <mesh key={idx} position={s.center}>
          <sphereGeometry args={[s.radius, 32, 32]} />
          <meshStandardMaterial
            color={new THREE.Color(s.color[0], s.color[1], s.color[2])}
            wireframe
          />
        </mesh>
      ))}
    </>
  );
}

export default function Home() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isInteracting, setIsInteracting] = useState(false);

  // State for rendering, though we don't show controls anymore
  const [lightPos] = useState<[number, number, number]>([5.0, 10.0, -5.0]);
  const [lightIntensity] = useState<number>(1.5);

  const [spheres] = useState<Sphere[]>([
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

  const fetchRender = async (cameraOrigin: number[], cameraTarget: number[]) => {
    setLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/render`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          spheres: spheres,
          light_pos: lightPos,
          light_intensity: lightIntensity,
          camera_origin: cameraOrigin,
          camera_target: cameraTarget
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

  const handleInteractionStart = () => {
    setIsInteracting(true);
  };

  const handleInteractionEnd = (camera: THREE.Camera, target: THREE.Vector3) => {
    setIsInteracting(false);

    // Convert camera position and target to arrays for API
    const origin = [camera.position.x, camera.position.y, camera.position.z];
    const targetArr = [target.x, target.y, target.z];

    fetchRender(origin, targetArr);
  };

  // Initial render on mount with default camera
  useEffect(() => {
    fetchRender([0.0, 0.8, -0.5], [0.0, 0.5, -3.0]);
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', overflow: 'hidden', backgroundColor: '#111' }}>

      {/* 3D Wireframe Scene (Always active for interaction, but obscured by image when idle) */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 1 }}>
        <Canvas>
          <Suspense fallback={null}>
            <SceneContent
              spheres={spheres}
              onInteractionStart={handleInteractionStart}
              onInteractionEnd={handleInteractionEnd}
            />
          </Suspense>
        </Canvas>
      </div>

      {/* HQ Render Overlay */}
      {imageUrl && !isInteracting && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: 2,
            pointerEvents: 'none', // Allow clicks to pass through to Canvas
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundImage: `url(${imageUrl})`,
            backgroundSize: 'cover', // Or contain? 'cover' essentially fills the screen
            backgroundPosition: 'center',
          }}
        >
          {loading && (
            <div style={{
              position: 'absolute',
              top: '20px',
              right: '20px',
              background: 'rgba(0,0,0,0.7)',
              color: 'white',
              padding: '10px',
              borderRadius: '4px'
            }}>
              Rendering...
            </div>
          )}
        </div>
      )}

      {/* Loading indicator when first loading or rendering without previous image */}
      {loading && !imageUrl && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'white',
          zIndex: 3
        }}>
          Rendering initial scene...
        </div>
      )}
    </div>
  );
}
