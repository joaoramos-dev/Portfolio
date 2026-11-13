import React, { Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { ContactShadows, Environment, Html, Lightformer, useGLTF } from '@react-three/drei'
import { ACESFilmicToneMapping, BoxGeometry, CanvasTexture, Color, LinearFilter, LinearMipmapLinearFilter, NoColorSpace, Object3D, ShaderMaterial, SRGBColorSpace, Vector3 } from 'three'
import { Bloom, EffectComposer, Vignette } from '@react-three/postprocessing'
import { gsap } from 'gsap'

const DESK_MODEL_URL = '/models/desk_room_fixed.glb?rev=crt-rebuild-20261110-1'
const CRT_PC_MODEL_URL = '/models/crt_pc_rebuilt.glb?rev=rebuilt-20261110-2'
const CRT_YAW = -0.2688
const CRT_PIVOT = [0.08, 0.476]
const rotateCrtPoint = ([x, y, z]) => {
  const offsetX = x - CRT_PIVOT[0]
  const offsetZ = z - CRT_PIVOT[1]
  const cosine = Math.cos(CRT_YAW)
  const sine = Math.sin(CRT_YAW)
  return [
    CRT_PIVOT[0] + offsetX * cosine + offsetZ * sine,
    y,
    CRT_PIVOT[1] - offsetX * sine + offsetZ * cosine,
  ]
}

const CAMERA_VIEWS = {
  overview: { position: [3.15, 1.75, 1.15], target: [0.18, 0.88, 1.16] },
  computer: { position: [1.65, 1.2, 0.93], target: rotateCrtPoint([0.308, 1.205, 0.476]) },
  mural: { position: [0.78, 1.34, 1.39], target: [-0.12, 1.31, 1.38] },
  photo: { position: [0.50, 0.87, 1.19], target: [-0.034, 0.82, 1.116] },
}

const FOCUS_DETAILS = {
  mural: { title: 'Sobre mim & competências', text: 'Engenharia de software, inteligência artificial e engenharia de dados. Explore as anotações e projetos no mural.' },
  photo: { title: 'Vamos conversar', text: 'Estou aberto a novos desafios e colaborações. Use o terminal do computador para navegar pelos projetos e formas de contato.' },
}

const LAMP_CYCLE_SECONDS = 40
const LAMP_FLICKER_KEYS = [
  [0, 0.1], [0.12, 1], [0.18, 0.18], [0.23, 0.9], [0.27, 0.05],
  [0.4, 1], [0.47, 0.56], [0.54, 0.95], [0.7, 0.93],
  [10.25, 0.9], [10.32, 0.08], [10.44, 1], [10.51, 0.5], [10.61, 0.95],
  [20.3, 0.95], [20.38, 0.05], [20.5, 1], [20.58, 0.28], [20.7, 0.94],
  [30.4, 0.95], [30.47, 0.06], [30.59, 0.9], [30.67, 0.35], [30.8, 0.95],
  [39.5, 0.95], [39.56, 0.04], [39.68, 0.9], [39.77, 0.25], [39.86, 0.95],
  [39.94, 0.1], [LAMP_CYCLE_SECONDS, 0.1],
]

function getLampFlicker(time) {
  const cycleTime = time % LAMP_CYCLE_SECONDS
  for (let index = 1; index < LAMP_FLICKER_KEYS.length; index += 1) {
    const [endTime, endIntensity] = LAMP_FLICKER_KEYS[index]
    if (cycleTime > endTime) continue
    const [startTime, startIntensity] = LAMP_FLICKER_KEYS[index - 1]
    const progress = (cycleTime - startTime) / (endTime - startTime)
    return startIntensity + (endIntensity - startIntensity) * progress
  }
  return LAMP_FLICKER_KEYS[0][1]
}

const CIGARETTE_SMOKE_VERTEX_SHADER = `
  varying vec3 vWorldPosition;
  void main() {
    vec3 worldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
    vWorldPosition = worldPosition;
    gl_Position = projectionMatrix * viewMatrix * vec4(worldPosition, 1.0);
  }
`

const CIGARETTE_SMOKE_FRAGMENT_SHADER = `
  uniform float uTime;
  uniform float uOpacity;
  uniform vec3 uCameraPosition;
  uniform vec3 uBoundsMin;
  uniform vec3 uBoundsMax;
  uniform vec3 uColor;
  varying vec3 vWorldPosition;

  float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
  }

  float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash31(i);
    float b = hash31(i + vec3(1.0, 0.0, 0.0));
    float c = hash31(i + vec3(0.0, 1.0, 0.0));
    float d = hash31(i + vec3(1.0, 1.0, 0.0));
    float e = hash31(i + vec3(0.0, 0.0, 1.0));
    float f1 = hash31(i + vec3(1.0, 0.0, 1.0));
    float g = hash31(i + vec3(0.0, 1.0, 1.0));
    float h = hash31(i + vec3(1.0, 1.0, 1.0));
    return mix(
      mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
      mix(mix(e, f1, f.x), mix(g, h, f.x), f.y),
      f.z
    );
  }

  void main() {
    vec3 rayDirection = normalize(vWorldPosition - uCameraPosition);
    vec3 inverseDirection = 1.0 / rayDirection;
    vec3 firstIntersection = (uBoundsMin - uCameraPosition) * inverseDirection;
    vec3 secondIntersection = (uBoundsMax - uCameraPosition) * inverseDirection;
    vec3 nearPlane = min(firstIntersection, secondIntersection);
    vec3 farPlane = max(firstIntersection, secondIntersection);
    float rayStart = max(max(nearPlane.x, nearPlane.y), nearPlane.z);
    float rayEnd = min(min(farPlane.x, farPlane.y), farPlane.z);
    if (rayEnd <= rayStart) discard;

    const int STEPS = 24;
    float stepLength = (rayEnd - rayStart) / float(STEPS);
    float accumulatedAlpha = 0.0;
    vec3 accumulatedColor = vec3(0.0);

    for (int step = 0; step < STEPS; step++) {
      float distanceAlongRay = rayStart + (float(step) + 0.5) * stepLength;
      vec3 worldPosition = uCameraPosition + rayDirection * distanceAlongRay;
      vec3 localPosition = (worldPosition - uBoundsMin) / (uBoundsMax - uBoundsMin);
      float height = localPosition.y;

      vec2 plumeCenter = vec2(0.5);
      plumeCenter.x += (noise3(vec3(height * 3.8 + 2.3, 3.1, uTime * 0.09)) - 0.5) * height * 0.9;
      plumeCenter.y += (noise3(vec3(7.4, height * 3.2 + 1.2, uTime * 0.08)) - 0.5) * height * 0.9;
      plumeCenter.x += (noise3(vec3(height * 10.0, uTime * 0.16 + 4.0, 3.7)) - 0.5) * height * 0.28;
      plumeCenter.y += (noise3(vec3(5.3, height * 9.0, uTime * 0.14 + 8.1)) - 0.5) * height * 0.28;
      float plumeRadius = 0.055 + pow(height, 0.82) * 0.33;
      vec3 boundaryCoord = vec3(localPosition.x * 12.0 + uTime * 0.035, height * 8.0 - uTime * 0.1, localPosition.z * 12.0 - uTime * 0.03);
      float boundaryVariation = noise3(boundaryCoord) - 0.5;
      float radialDistance = length((localPosition.xz - plumeCenter) * vec2(1.0, 1.08));
      float softBoundary = 1.0 - smoothstep(plumeRadius * (0.52 + boundaryVariation * 0.42), plumeRadius * (1.14 + boundaryVariation * 0.5), radialDistance);

      vec3 flow = vec3(
        (localPosition.x - plumeCenter.x) * 9.0,
        height * 6.0,
        (localPosition.z - plumeCenter.y) * 9.0
      );
      flow -= vec3(uTime * 0.12, uTime * 0.24, -uTime * 0.09);
      vec3 warp = vec3(
        noise3(flow * 0.72 + vec3(uTime * 0.055, 2.1, 4.7)),
        noise3(flow * 0.72 + vec3(8.3, uTime * 0.045, 1.6)),
        noise3(flow * 0.72 + vec3(3.2, 6.4, uTime * 0.05))
      ) - 0.5;
      vec3 warpedFlow = flow + warp * (0.9 + height * 1.1);
      float broadDensity = noise3(warpedFlow * 1.15);
      float midDensity = noise3(warpedFlow * 2.7 + vec3(5.2, 2.8, 7.1));
      float fineDensity = noise3(warpedFlow * 5.8 + vec3(8.1, 1.3, 3.4));
      float wispyDensity = smoothstep(0.31 + height * 0.05, 0.68, broadDensity * 0.55 + midDensity * 0.3 + fineDensity * 0.15);

      float emission = smoothstep(0.0, 0.035, height) * exp(-height * 1.7) * (1.0 - smoothstep(0.76, 1.0, height));
      float density = wispyDensity * softBoundary * emission * (1.1 + midDensity * 0.65);
      float sampleAlpha = 1.0 - exp(-density * uOpacity * stepLength * 7.5);
      float transmission = 1.0 - accumulatedAlpha;
      float softLighting = 0.78 + broadDensity * 0.18 + (1.0 - height) * 0.08;
      accumulatedColor += uColor * softLighting * sampleAlpha * transmission;
      accumulatedAlpha += transmission * sampleAlpha;
      if (accumulatedAlpha > 0.98) break;
    }

    gl_FragColor = vec4(accumulatedColor / max(accumulatedAlpha, 0.0001), accumulatedAlpha);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`

function SmokePlume({ position, rise = 0.42, width = 0.16, opacity = 0.9, color = '#c2beb8' }) {
  const { camera } = useThree()
  const plume = useRef(null)
  const boundsMin = useMemo(() => new Vector3(), [])
  const boundsMax = useMemo(() => new Vector3(), [])
  const geometry = useMemo(() => new BoxGeometry(1, 1, 1), [])
  const material = useMemo(() => new ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uOpacity: { value: opacity },
      uCameraPosition: { value: new Vector3() },
      uBoundsMin: { value: new Vector3() },
      uBoundsMax: { value: new Vector3() },
      uColor: { value: new Color(color) },
    },
    vertexShader: CIGARETTE_SMOKE_VERTEX_SHADER,
    fragmentShader: CIGARETTE_SMOKE_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
  }), [])

  useEffect(() => () => {
    geometry.dispose()
    material.dispose()
  }, [geometry, material])

  useFrame(({ clock }) => {
    if (!plume.current) return
    boundsMin.set(position[0] - width / 2, position[1], position[2] - width / 2)
    boundsMax.set(position[0] + width / 2, position[1] + rise, position[2] + width / 2)
    camera.getWorldPosition(material.uniforms.uCameraPosition.value)
    material.uniforms.uBoundsMin.value.copy(boundsMin)
    material.uniforms.uBoundsMax.value.copy(boundsMax)
    material.uniforms.uTime.value = clock.elapsedTime
  })

  return <mesh
    ref={plume}
    geometry={geometry}
    material={material}
    position={[position[0], position[1] + rise / 2, position[2]]}
    scale={[width, rise, width]}
    frustumCulled={false}
  />
}

function CameraDirector({ focus }) {
  const { camera } = useThree()
  const target = useRef(new Vector3(...CAMERA_VIEWS.overview.target))
  useLayoutEffect(() => {
    const view = CAMERA_VIEWS[focus] || CAMERA_VIEWS.overview
    if (focus === 'overview' && camera.position.distanceTo(new Vector3(...view.position)) < 0.001) {
      camera.lookAt(...view.target)
      return undefined
    }
    gsap.killTweensOf(camera.position)
    gsap.killTweensOf(target.current)
    const positionTween = gsap.to(camera.position, { x: view.position[0], y: view.position[1], z: view.position[2], duration: 1.9, ease: 'power3.inOut' })
    const targetTween = gsap.to(target.current, { x: view.target[0], y: view.target[1], z: view.target[2], duration: 1.9, ease: 'power3.inOut', onUpdate: () => camera.lookAt(target.current), onComplete: () => camera.lookAt(target.current) })
    return () => { positionTween.kill(); targetTween.kill() }
  }, [camera, focus])
  return null
}

function Hotspot({ position, size, focus, onFocus, onHover }) {
  useEffect(() => () => { document.body.style.cursor = '' }, [])
  return <mesh position={position} onClick={(event) => { event.stopPropagation(); onFocus(focus) }} onPointerOver={(event) => { event.stopPropagation(); document.body.style.cursor = 'pointer'; onHover(focus) }} onPointerOut={() => { document.body.style.cursor = ''; onHover(null) }}>
    <boxGeometry args={size} />
    <meshBasicMaterial transparent opacity={0} colorWrite={false} depthWrite={false} />
  </mesh>
}

function DeskLampLight() {
  const target = useMemo(() => new Object3D(), [])
  const light = useRef(null)
  // Match the filament center modeled in the Blender shade-local coordinates.
  const bulbPosition = [0.18235, 1.63836, 1.81426]
  const nozzleAxis = new Vector3(0.1044, -0.8716, -0.4789).normalize()
  const shadeScale = 1.3
  target.position.set(
    bulbPosition[0] + nozzleAxis.x * 0.884 * shadeScale,
    bulbPosition[1] + nozzleAxis.y * 0.884 * shadeScale,
    bulbPosition[2] + nozzleAxis.z * 0.884 * shadeScale,
  )
  useFrame(({ clock }) => {
    const flicker = getLampFlicker(clock.elapsedTime)
    if (light.current) light.current.intensity = 82 * flicker
  })
  return <>
    <primitive object={target} />
    {/* The bulb and filament are modeled inside the exported Blender shade. */}
    <spotLight ref={light} position={bulbPosition} target={target} intensity={82 * LAMP_FLICKER_KEYS[0][1]} angle={0.82} penumbra={0.82} distance={3.6} decay={2} color="#ffb85f" castShadow shadow-mapSize={[1024, 1024]} shadow-camera-near={0.01} shadow-camera-far={3.6} shadow-normalBias={0.01} />
  </>
}

function createPinBoardTexture() {
  const canvas = document.createElement('canvas')
  canvas.width = 1200
  canvas.height = 800
  const ctx = canvas.getContext('2d')
  let seed = 0x47a39d
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 0x100000000
  }

  ctx.fillStyle = '#38251a'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  const vignette = ctx.createRadialGradient(580, 340, 90, 600, 400, 760)
  vignette.addColorStop(0, 'rgba(132, 82, 43, 0.35)')
  vignette.addColorStop(1, 'rgba(8, 5, 4, 0.25)')
  ctx.fillStyle = vignette
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  for (let i = 0; i < 15000; i += 1) {
    const x = random() * canvas.width
    const y = random() * canvas.height
    const radius = 0.4 + random() * 1.8
    ctx.fillStyle = random() > 0.48 ? 'rgba(197, 142, 85, 0.10)' : 'rgba(12, 8, 6, 0.14)'
    ctx.fillRect(x, y, radius, radius)
  }

  const drawPin = (x, y, color = '#8c3323') => {
    const pin = ctx.createRadialGradient(x - 3, y - 3, 1, x, y, 11)
    pin.addColorStop(0, '#f0ba79')
    pin.addColorStop(0.26, color)
    pin.addColorStop(1, '#24130e')
    ctx.fillStyle = pin
    ctx.beginPath()
    ctx.arc(x, y, 9, 0, Math.PI * 2)
    ctx.fill()
  }

  // Yellow planning note on the left side of the board.
  ctx.save()
  ctx.translate(390, 208)
  ctx.rotate(-0.035)
  ctx.shadowColor = 'rgba(0, 0, 0, 0.55)'
  ctx.shadowBlur = 22
  ctx.shadowOffsetY = 12
  ctx.fillStyle = '#b88736'
  ctx.fillRect(0, 0, 190, 190)
  ctx.shadowColor = 'transparent'
  ctx.fillStyle = 'rgba(245, 202, 111, 0.20)'
  ctx.fillRect(0, 0, 190, 8)
  ctx.fillStyle = '#392818'
  ctx.textAlign = 'left'
  ctx.font = 'bold 24px monospace'
  ctx.fillText('SONHOS', 16, 50)
  ctx.font = '19px monospace'
  ctx.fillText('PLANEJAMENTO', 16, 86)
  ctx.fillText('EXECUÇÃO', 16, 120)
  ctx.font = '28px monospace'
  ctx.fillText('→', 72, 158)
  drawPin(94, 8, '#6c361f')
  ctx.restore()

  // Project checklist at upper right, with paper grain and a slight curl.
  ctx.save()
  ctx.translate(650, 108)
  ctx.rotate(-0.018)
  ctx.shadowColor = 'rgba(0, 0, 0, 0.6)'
  ctx.shadowBlur = 24
  ctx.shadowOffsetY = 13
  ctx.fillStyle = '#bdb2a0'
  ctx.fillRect(0, 0, 390, 300)
  ctx.shadowColor = 'transparent'
  ctx.fillStyle = 'rgba(68, 52, 39, 0.12)'
  for (let i = 0; i < 450; i += 1) ctx.fillRect(random() * 390, random() * 300, 1 + random() * 3, 1)
  ctx.textAlign = 'center'
  ctx.fillStyle = '#3e352d'
  ctx.font = 'bold 25px monospace'
  ctx.fillText('PROJETOS', 195, 42)
  ctx.strokeStyle = 'rgba(68, 52, 39, 0.35)'
  ctx.beginPath()
  ctx.moveTo(22, 57)
  ctx.lineTo(368, 57)
  ctx.stroke()
  ctx.textAlign = 'left'
  ctx.font = '18px monospace'
  const checklist = ['Ogma Workstation', 'Replicação (IA + Segurança)', 'Portfólio Web', 'Próximos passos...']
  checklist.forEach((item, index) => {
    const y = 96 + index * 43
    ctx.strokeStyle = '#40392f'
    ctx.lineWidth = 2
    ctx.strokeRect(25, y - 17, 15, 15)
    ctx.fillStyle = '#3d342b'
    ctx.fillText(item, 54, y - 3)
  })
  drawPin(195, 6, '#8e3828')
  ctx.restore()

  // A small sunset photograph near the lower middle of the board.
  ctx.save()
  ctx.translate(475, 510)
  ctx.rotate(-0.075)
  ctx.shadowColor = 'rgba(0, 0, 0, 0.65)'
  ctx.shadowBlur = 20
  ctx.shadowOffsetY = 13
  ctx.fillStyle = '#d1c5ad'
  ctx.fillRect(0, 0, 310, 205)
  ctx.shadowColor = 'transparent'
  const sky = ctx.createLinearGradient(0, 10, 0, 158)
  sky.addColorStop(0, '#685042')
  sky.addColorStop(0.48, '#db995b')
  sky.addColorStop(0.76, '#f2bd79')
  sky.addColorStop(1, '#675343')
  ctx.fillStyle = sky
  ctx.fillRect(13, 12, 284, 150)
  ctx.fillStyle = '#6a4c36'
  ctx.beginPath()
  ctx.moveTo(13, 123); ctx.lineTo(68, 85); ctx.lineTo(116, 113); ctx.lineTo(166, 74)
  ctx.lineTo(225, 113); ctx.lineTo(297, 87); ctx.lineTo(297, 162); ctx.lineTo(13, 162); ctx.closePath(); ctx.fill()
  ctx.fillStyle = '#37281f'
  ctx.beginPath()
  ctx.moveTo(13, 145); ctx.lineTo(84, 125); ctx.lineTo(148, 143); ctx.lineTo(215, 118)
  ctx.lineTo(297, 139); ctx.lineTo(297, 162); ctx.lineTo(13, 162); ctx.closePath(); ctx.fill()
  ctx.fillStyle = '#211b17'
  ctx.beginPath()
  ctx.ellipse(205, 136, 13, 24, -0.08, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillRect(199, 150, 12, 10)
  ctx.strokeStyle = '#211b17'
  ctx.lineWidth = 7
  ctx.beginPath(); ctx.moveTo(203, 158); ctx.lineTo(192, 162); ctx.moveTo(207, 158); ctx.lineTo(216, 162); ctx.stroke()
  drawPin(154, 7, '#a33b27')
  ctx.restore()

  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  texture.anisotropy = 8
  return texture
}

function PinBoardArtwork() {
  const texture = useMemo(createPinBoardTexture, [])
  useEffect(() => () => texture.dispose(), [texture])
  return <mesh position={[-0.091, 1.301, 1.384]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
    <planeGeometry args={[0.84, 0.56]} />
    <meshStandardMaterial map={texture} roughness={0.96} metalness={0} />
  </mesh>
}

function MonitorScreen() {
  const [page, setPage] = useState('home')
  const [command, setCommand] = useState('')
  const pages = {
    home: { command: 'help', rows: ['Olá, eu sou João Pedro Ramos.', 'Engenheiro de Software • IA • Dados', 'Escolha um comando para explorar o portfólio.'] },
    sobre: { command: 'cat sobre.txt', rows: ['Engenheiro de Software e IA.', 'Experiência com sistemas, modelos inteligentes', 'e soluções orientadas a dados.'] },
    projetos: { command: 'ls ./projetos', rows: ['ogma-workstation/', 'rag-seguranca/', 'portfolio-web/'] },
    stack: { command: 'cat ./skills.json', rows: ['Software: React, TypeScript, Python', 'IA: LLMs, RAG, avaliação de modelos', 'Dados: pipelines, SQL, engenharia de dados'] },
    contato: { command: 'cat contato.txt', rows: ['Vamos conversar sobre seu próximo projeto.', 'Contato profissional disponível.'] },
  }
  const current = pages[page]
  const runCommand = (value) => {
    const normalized = value.trim().toLowerCase()
    const aliases = { help: 'home', inicio: 'home', home: 'home', sobre: 'sobre', about: 'sobre', projetos: 'projetos', projects: 'projetos', skills: 'stack', tecnologias: 'stack', contato: 'contato', contact: 'contato' }
    const next = Object.keys(aliases).find((alias) => normalized === alias || normalized.includes(alias))
    if (next) setPage(aliases[next])
    setCommand('')
  }
  // The terminal surface sits flush over the CRT glass, facing the camera.
  return <Html transform position={rotateCrtPoint([0.308, 1.205, 0.476])} rotation={[0, Math.PI / 2 + CRT_YAW, 0]} distanceFactor={0.5} pointerEvents="auto" occlude={false} zIndexRange={[16777271, 0]}>
    <div className="monitor-terminal" onPointerDown={(event) => event.stopPropagation()}>
      <div className="terminal-titlebar"><span><i/><i/><i/></span><small>joaoramos@dev: ~</small><b>⌄</b></div>
      <div className="terminal-output" aria-live="polite">
        <div><span className="terminal-user">joaoramos@dev</span>:<span className="terminal-path">~</span>$ {current.command}</div>
        {current.rows.map((row) => <p key={row}>{row}</p>)}
        <div className="terminal-links">{Object.entries({ sobre: 'sobre', projetos: 'projetos', stack: 'skills', contato: 'contato' }).map(([key, label]) => <button key={key} onClick={() => setPage(key)}>{label}</button>)}</div>
        <form className="terminal-prompt" onSubmit={(event) => { event.preventDefault(); runCommand(command) }}><span className="terminal-user">joaoramos@dev</span>:<span className="terminal-path">~</span>$ <input aria-label="Comando do terminal" value={command} onChange={(event) => setCommand(event.target.value)} autoComplete="off" spellCheck="false"/><span className="terminal-caret"/></form>
      </div>
    </div>
  </Html>
}
function AssetModel({ onFocus, onHover, hoveredObject }) {
  const { scene } = useGLTF(DESK_MODEL_URL)
  const { scene: crtScene } = useGLTF(CRT_PC_MODEL_URL)
  const gl = useThree((state) => state.gl)
  const model = useMemo(() => {
    const clone = scene.clone(true)
    for (const name of ['Monitor', 'Computer']) {
      const oldComputerPart = clone.getObjectByName(name)
      if (oldComputerPart) oldComputerPart.visible = false
    }
    const anisotropy = Math.min(16, gl.capabilities.getMaxAnisotropy())
    clone.traverse((node) => {
      if (!node.isMesh) return
      const isLampBulbPart = node.name.startsWith('Portfolio_Lamp_Bulb_') || node.name === 'Portfolio_Lamp_Filament'
      const isMonitor = node.name === 'Monitor'
      node.castShadow = !isLampBulbPart
      node.receiveShadow = !isLampBulbPart && !isMonitor
      node.material = Array.isArray(node.material)
        ? node.material.map((material) => material.clone())
        : node.material.clone()
      const materials = Array.isArray(node.material) ? node.material : [node.material]
      for (const material of materials) {
        if (isMonitor && material.normalMap) {
          material.normalMap = null
          material.needsUpdate = true
        }
        if (['Computer', 'Monitor', 'Keyboard', 'Computer_Mouse'].includes(node.name) && material.name === 'M_Computer_2048') {
          material.color.set('#d3b993')
        }
        for (const key of ['map', 'emissiveMap', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap']) {
          const texture = material[key]
          if (!texture) continue
          texture.colorSpace = key === 'map' || key === 'emissiveMap' ? SRGBColorSpace : NoColorSpace
          texture.generateMipmaps = true
          texture.minFilter = LinearMipmapLinearFilter
          texture.magFilter = LinearFilter
          if (texture.anisotropy !== anisotropy) {
            texture.anisotropy = anisotropy
            texture.needsUpdate = true
          }
        }
      }
    })
    return clone
  }, [scene, gl])
  const crtModel = useMemo(() => {
    const clone = crtScene.clone(true)
    clone.traverse((node) => {
      if (!node.isMesh) return
      node.castShadow = true
      node.receiveShadow = true
      node.material = Array.isArray(node.material)
        ? node.material.map((material) => material.clone())
        : node.material.clone()
    })
    return clone
  }, [crtScene])
  useEffect(() => {
    if (!hoveredObject) return undefined
    const roots = Array.isArray(hoveredObject) ? hoveredObject : [hoveredObject]
    const previous = []
    roots.forEach((root) => root?.traverse((node) => {
      if (!node.isMesh) return
      const materials = Array.isArray(node.material) ? node.material : [node.material]
      materials.forEach((material) => {
        if (!material.emissive) return
        previous.push({ material, color: material.emissive.clone(), intensity: material.emissiveIntensity })
        material.emissive.set('#c2a66d')
        material.emissiveIntensity = 0.28
      })
    }))
    return () => previous.forEach(({ material, color, intensity }) => {
      material.emissive.copy(color)
      material.emissiveIntensity = intensity
    })
  }, [hoveredObject])
  const focusObjects = useMemo(() => {
    const mural = model.getObjectByName('PinBoard')
    const muralMeshes = []
    const computerMeshes = []
    mural?.traverse((node) => { if (node.isMesh) muralMeshes.push(node) })
    crtModel.traverse((node) => { if (node.isMesh) computerMeshes.push(node) })
    return {
      computer: computerMeshes,
      mural: muralMeshes,
      photo: model.getObjectByName('PhotoFrame_10x15'),
    }
  }, [model, crtModel])
  return <>
    <primitive object={model} />
    <group position={[CRT_PIVOT[0], 0, CRT_PIVOT[1]]} rotation={[0, CRT_YAW, 0]}>
      <primitive object={crtModel} position={[-CRT_PIVOT[0], 0, -CRT_PIVOT[1]]} />
    </group>
    <Hotspot position={[0.08, 1.12, 0.476]} size={[0.7, 0.74, 0.72]} focus="computer" onFocus={onFocus} onHover={(key) => onHover(key ? focusObjects[key] : null)} />
    <Hotspot position={[-0.12, 1.31, 1.38]} size={[0.09, 0.62, 0.82]} focus="mural" onFocus={onFocus} onHover={(key) => onHover(key ? focusObjects[key] : null)} />
    <Hotspot position={[-0.03, 0.82, 1.12]} size={[0.24, 0.27, 0.14]} focus="photo" onFocus={onFocus} onHover={(key) => onHover(key ? focusObjects[key] : null)} />
  </>
}

function DeskFallback() {
  return <group>
    <mesh position={[0.2, 0.72, 1.17]} castShadow receiveShadow><boxGeometry args={[0.86, 0.08, 2.02]} /><meshStandardMaterial color="#59412e" roughness={0.78} /></mesh>
    <mesh position={[0.1, 1.05, 0.5]} castShadow><boxGeometry args={[0.58, 0.38, 0.045]} /><meshStandardMaterial color="#242627" roughness={0.35} metalness={0.25} /></mesh>
    <mesh position={[0.1, 1.27, 1.38]} castShadow><boxGeometry args={[0.74, 0.52, 0.06]} /><meshStandardMaterial color="#493a2f" roughness={0.9} /></mesh>
    {[[-0.2, 0.36, 0.32], [0.58, 0.36, 0.32], [-0.2, 0.36, 2.02], [0.58, 0.36, 2.02]].map(([x, y, z]) => <mesh key={`${x}-${z}`} position={[x, y, z]} castShadow><boxGeometry args={[0.055, 0.72, 0.055]} /><meshStandardMaterial color="#342920" roughness={0.78} /></mesh>)}
  </group>
}

export default function DeskScene() {
  const [focus, setFocus] = useState('overview')
  const [hoveredObject, setHoveredObject] = useState(null)
  useEffect(() => {
    const handleKey = (event) => { if (event.key === 'Escape') setFocus('overview') }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [])
  const detail = FOCUS_DETAILS[focus] || (focus === 'computer' ? { title: 'Terminal de projetos', text: 'Explore o terminal Linux na tela do computador. Digite “help” para listar os comandos.' } : null)
  return <div className="desk-scene">
    <Canvas shadows dpr={[1, 1.5]} camera={{ position: CAMERA_VIEWS.overview.position, fov: 34, near: 0.1, far: 40 }} gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }} onCreated={({ gl }) => { gl.outputColorSpace = SRGBColorSpace; gl.toneMapping = ACESFilmicToneMapping; gl.toneMappingExposure = 1 }}>
      <color attach="background" args={['#100b08']} />
      <fog attach="fog" args={['#100b08', 8, 18]} />
      <ambientLight intensity={0.08} color="#d9bd99" />
      <directionalLight position={[-2, 4, 2]} intensity={0.12} color="#dfc9aa" />
      <directionalLight position={[3, 3, -2]} intensity={0.015} color="#a39d88" />
      <pointLight position={[0.48, 1.22, 0.58]} intensity={0.025} distance={0.8} decay={2} color="#74c6a2" />
      <DeskLampLight />
      <Environment resolution={128} background={false} environmentIntensity={0.12}>
        <Lightformer form="rect" intensity={0.8} color="#ffddb3" position={[-2, 3, 2]} scale={[4, 4, 1]} />
        <Lightformer form="rect" intensity={0.06} color="#c1ad89" position={[3, 2, -1]} scale={[3, 3, 1]} />
      </Environment>
      <mesh position={[-0.2, 1.42, 1.16]} rotation={[0, Math.PI / 2, 0]} receiveShadow><planeGeometry args={[3.6, 2.9]} /><meshStandardMaterial color="#211813" roughness={0.97} /></mesh>
      <mesh position={[0, 0.006, 0.75]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow><planeGeometry args={[200, 200]} /><meshStandardMaterial color="#100d0b" roughness={0.96} /></mesh>
      <CameraDirector focus={focus} />
      <ContactShadows position={[0, 0.006, 0.8]} opacity={0.42} scale={4} blur={2.2} far={2} />
      <Suspense fallback={<DeskFallback />}><AssetModel onFocus={setFocus} onHover={setHoveredObject} hoveredObject={hoveredObject} /></Suspense>
      <SmokePlume position={[-0.034, 0.819, 1.436]} rise={0.36} width={0.15} opacity={0.76} color="#c2beb8" />
      <SmokePlume position={[0.49, 0.78, 1.61]} />
      <MonitorScreen />
      <EffectComposer multisampling={2}><Bloom intensity={0.2} luminanceThreshold={1.1} luminanceSmoothing={0.6} /><Vignette eskil={false} offset={0.2} darkness={0.68} /></EffectComposer>
    </Canvas>
    {focus !== 'overview' && <aside className="focus-panel" aria-live="polite">
      <div><span className="focus-kicker">PONTO DE INVESTIGAÇÃO</span><h2>{detail?.title}</h2><p>{detail?.text}</p></div>
      <button onClick={() => setFocus('overview')}>← Voltar à mesa <kbd>Esc</kbd></button>
    </aside>}
  </div>
}

useGLTF.preload(DESK_MODEL_URL)
useGLTF.preload(CRT_PC_MODEL_URL)
