import { useState, useEffect } from 'react'
import './App.css'

function App() {
  // Simulation state
  const [speed, setSpeed] = useState(84)
  const [steeringAngle, setSteeringAngle] = useState(-12.5)
  const [brake, setBrake] = useState(15)
  const [throttle, setThrottle] = useState(68)
  const [maxSpeed] = useState(240)
  const [avgSpeed] = useState(62)

  // ============================================================
  // CONNECT YOUR AI MODEL HERE - Replace this entire useEffect
  // ============================================================
  useEffect(() => {
    // OPTION 1: WebSocket Connection (Real-time streaming)
    // ------------------------------------------------------------
    // const ws = new WebSocket('ws://your-ai-model-server:8080/stream');
    // 
    // ws.onmessage = (event) => {
    //   const aiOutput = JSON.parse(event.data);
    //   setSpeed(aiOutput.speed);              // km/h (0-240)
    //   setSteeringAngle(aiOutput.steering);   // degrees (-90 to +90)
    //   setBrake(aiOutput.brake);              // percentage (0-100)
    //   setThrottle(aiOutput.throttle);        // percentage (0-100)
    // };
    //
    // return () => ws.close();

    // OPTION 2: HTTP Polling (Fetch data every interval)
    // ------------------------------------------------------------
    // const interval = setInterval(async () => {
    //   try {
    //     const response = await fetch('http://your-ai-model-server:8080/predict');
    //     const aiOutput = await response.json();
    //     
    //     setSpeed(aiOutput.speed);              // km/h (0-240)
    //     setSteeringAngle(aiOutput.steering);   // degrees (-90 to +90)
    //     setBrake(aiOutput.brake);              // percentage (0-100)
    //     setThrottle(aiOutput.throttle);        // percentage (0-100)
    //   } catch (error) {
    //     console.error('Error fetching AI predictions:', error);
    //   }
    // }, 100); // Update every 100ms
    //
    // return () => clearInterval(interval);

    // OPTION 3: Direct API Integration with Python Backend
    // ------------------------------------------------------------
    // const interval = setInterval(async () => {
    //   try {
    //     const response = await fetch('http://localhost:5000/api/driving-data', {
    //       method: 'GET',
    //       headers: { 'Content-Type': 'application/json' }
    //     });
    //     const data = await response.json();
    //     
    //     // Map your AI model outputs to the dashboard
    //     setSpeed(data.predicted_speed || 0);
    //     setSteeringAngle(data.predicted_steering || 0);
    //     setBrake(data.predicted_brake || 0);
    //     setThrottle(data.predicted_throttle || 0);
    //   } catch (error) {
    //     console.error('AI Model connection error:', error);
    //   }
    // }, 100);
    //
    // return () => clearInterval(interval);

    // TEMPORARY: Simulation mode (Remove when connecting real AI model)
    // ------------------------------------------------------------
    const interval = setInterval(() => {
      // HOW IT WORKS: Values vary every 100 milliseconds (10 times per second)
      // 
      // setInterval() creates a timer that repeatedly executes code
      // The number 100 at the end means "run every 100ms"
      
      // SPEED: Changes by -2.5 to +2.5 km/h each update
      // Math.random() generates 0 to 1, subtract 0.5 = range of -0.5 to +0.5
      // Multiply by 5 = range of -2.5 to +2.5
      // Math.max(0, ...) ensures speed never goes below 0
      // Math.min(240, ...) ensures speed never exceeds 240 km/h
      setSpeed(prev => Math.max(0, Math.min(240, prev + (Math.random() - 0.5) * 5)))
      
      // STEERING: Changes by -1.5 to +1.5 degrees each update
      // Range: -90° (hard left) to +90° (hard right)
      setSteeringAngle(prev => Math.max(-90, Math.min(90, prev + (Math.random() - 0.5) * 3)))
      
      // BRAKE: Changes by -4 to +4 percentage each update
      // Range: 0% (no brake) to 100% (full brake)
      setBrake(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 8)))
      
      // THROTTLE/ACCELERATION: Changes by -4 to +4 percentage each update
      // Range: 0% (no throttle) to 100% (full throttle)
      setThrottle(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 8)))
      
      // WHEN YOU CONNECT YOUR AI MODEL:
      // Replace these random calculations with actual AI predictions
      // Example: setSpeed(aiOutput.speed) instead of the random formula
    }, 100) // ← This 100 means update every 100 milliseconds

    return () => clearInterval(interval)
  }, [])

  // Calculate gauge arc offset for speed gauge
  const circumference = 2 * Math.PI * 80
  const speedPercentage = speed / maxSpeed
  const speedOffset = circumference - (speedPercentage * (circumference * 0.75))

  return (
    <div className="bg-background-dark min-h-screen flex flex-col text-white selection:bg-primary selection:text-white">
      {/* Background Effects */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-grid-pattern bg-[length:40px_40px] opacity-20"></div>
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[128px]"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent-cyan/10 rounded-full blur-[128px]"></div>
      </div>

      {/* Header */}
      <header className="relative z-50 flex items-center justify-between border-b border-glass-border bg-surface-dark/80 backdrop-blur-md px-8 py-4">
        <div className="flex items-center gap-5">
          <div className="relative group">
            <div className="absolute inset-0 bg-primary/40 rounded-xl blur group-hover:blur-md transition-all duration-300"></div>
            <div className="relative size-10 flex items-center justify-center text-white bg-gradient-to-br from-primary to-blue-600 rounded-xl shadow-inner border border-white/20">
              <span className="material-symbols-outlined text-[24px]">speed</span>
            </div>
          </div>
          <div>
            <h2 className="text-xl font-bold font-display tracking-tight text-white">
              AI-Driven Driver  <span className="text-primary">Pro</span>
            </h2>
            <div className="flex items-center gap-2 text-xs font-mono text-gray-400 mt-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse"></span>
              <span>L1 LIVE</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-8">
          <div className="hidden md:flex items-center gap-6 text-sm font-medium text-gray-400 bg-surface-lighter/30 px-4 py-2 rounded-full border border-white/5">
            
            <div className="w-px h-3 bg-white/10"></div>
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[16px] text-primary">cloud_done</span>
              <span className="font-mono text-gray-300">Connected</span>
            </div>
          </div>
          <div className="relative">
            <div className="size-10 rounded-full p-0.5 bg-gradient-to-tr from-primary to-accent-cyan">
              <div 
                className="size-full bg-center bg-no-repeat bg-cover rounded-full border-2 border-surface-dark" 
                style={{backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuDb0wfAWsiXVVC17LSg5ejHKx3gSh_Ve7c9NS6cDY2m5kTxcWPtLCwxFTFhhGgGZeJ3oqxGyE9Nbezs3AIKMI1aTMfwad6hSCYSByLCPfiEC_ITnNIAU8jS9z0AKMgGVKRDhQTo7wK0QLmsMQ4J8WIXrB7GAOmclBJEtNAsOGlkv4W76YEae3xZqpDU0By4DAQrmMofnm3ZgRt3rYaySmXYq6KX-W1DYq2eeUNWCvdIejZHPi3u2mq4yHWyaC5oDSFmwZclHRCzi7jr")'}}
              ></div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-6 md:p-10 overflow-hidden">
        <div className="w-full max-w-[1400px] grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch h-full max-h-[800px]">
          
          {/* Speed Gauge */}
          <div className="lg:col-span-5 glass-panel rounded-2xl flex flex-col relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-6 opacity-50 group-hover:opacity-100 transition-opacity">
              <span className="material-symbols-outlined text-gray-500">settings_motion_mode</span>
            </div>
            <div className="flex-1 flex flex-col items-center justify-center p-8 relative">
              <div className="absolute inset-0 bg-radial-gradient from-primary/10 to-transparent opacity-50"></div>
              <h3 className="absolute top-6 left-8 text-gray-400 text-xs font-bold font-display uppercase tracking-[0.2em]">
                Vehicle Speed
              </h3>
              <div className="relative w-full max-w-[400px] aspect-square flex items-center justify-center">
                <div className="absolute inset-4 border border-white/5 rounded-full border-dashed animate-[spin_60s_linear_infinite]"></div>
                <svg className="w-full h-full -rotate-90 transform drop-shadow-2xl" viewBox="0 0 200 200">
                  <defs>
                    <linearGradient id="speedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" style={{stopColor: '#3b82f6', stopOpacity: 1}} />
                      <stop offset="100%" style={{stopColor: '#06b6d4', stopOpacity: 1}} />
                    </linearGradient>
                    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                      <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                  </defs>
                  <circle 
                    className="opacity-50" 
                    cx="100" 
                    cy="100" 
                    r="80" 
                    fill="none" 
                    stroke="#1e293b" 
                    strokeWidth="12" 
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={circumference * 0.25}
                  />
                  <circle 
                    className="gauge-arc" 
                    cx="100" 
                    cy="100" 
                    r="80" 
                    fill="none" 
                    stroke="url(#speedGradient)" 
                    strokeWidth="12" 
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={speedOffset}
                    filter="url(#glow)"
                  />
                  <g className="text-gray-600" stroke="currentColor" strokeWidth="2">
                    {[-135, -90, -45, 0, 45, 90, 135].map((angle, i) => (
                      <line 
                        key={i}
                        x1="100" 
                        y1="10" 
                        x2="100" 
                        y2="20"
                        transform={`rotate(${angle} 100 100)`}
                      />
                    ))}
                  </g>
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
                  <div className="text-8xl md:text-9xl font-bold font-display tracking-tighter text-white glow-text leading-none">
                    {Math.round(speed)}
                  </div>
                  <div className="text-sm font-mono text-accent-cyan tracking-[0.3em] font-medium mt-4 uppercase border border-accent-cyan/30 px-3 py-1 rounded bg-accent-cyan/5">
                    km/h
                  </div>
                </div>
              </div>
              <div className="mt-8 flex gap-8">
                <div className="flex flex-col items-center">
                  <span className="text-xs text-gray-500 font-mono uppercase">Max</span>
                  <span className="text-lg font-bold font-display text-gray-300">{maxSpeed}</span>
                </div>
                <div className="w-px h-10 bg-white/10"></div>
                <div className="flex flex-col items-center">
                  <span className="text-xs text-gray-500 font-mono uppercase">Avg</span>
                  <span className="text-lg font-bold font-display text-gray-300">{avgSpeed}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Steering Wheel */}
          <div className="lg:col-span-4 glass-panel rounded-2xl flex flex-col relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500/0 via-purple-500/50 to-purple-500/0"></div>
            <div className="p-6 flex justify-between items-start">
              <div>
                <h3 className="text-gray-400 text-xs font-bold font-display uppercase tracking-[0.2em] mb-1">
                  Steering Angle
                </h3>
                <div className="text-xs text-gray-500">Real-time input</div>
              </div>
              <div className="bg-surface-lighter/50 border border-white/10 px-3 py-1.5 rounded-lg flex items-center gap-2 shadow-lg">
                <span className="material-symbols-outlined text-purple-400 text-sm">rotate_left</span>
                <span className="text-white font-mono font-bold">{steeringAngle.toFixed(1)}°</span>
              </div>
            </div>
            <div className="flex-1 flex flex-col items-center justify-center p-4">
              <div className="relative size-64 flex items-center justify-center">
                <div className="absolute inset-0 bg-purple-500/10 rounded-full blur-2xl"></div>
                <div 
                  className="relative w-full h-full transition-transform duration-100 ease-out drop-shadow-2xl"
                  style={{transform: `rotate(${steeringAngle}deg)`}}
                >
                  <div className="absolute inset-0 rounded-full overflow-hidden z-10">
                    <div className="absolute inset-0 bg-surface-dark/30"></div>
                    <div className="absolute top-1/2 left-0 w-full h-14 -translate-y-1/2 flex items-center justify-between">
                      <div className="w-full h-full bg-gradient-to-b from-surface-lighter via-surface-dark to-surface-lighter shadow-lg flex items-center justify-between px-6 border-y border-white/5">
                        <div className="flex gap-1">
                          <div className="w-2 h-2 rounded-full bg-red-500/50 shadow-inner"></div>
                          <div className="w-2 h-2 rounded-full bg-white/20 shadow-inner"></div>
                        </div>
                        <div className="flex gap-1">
                          <div className="w-2 h-2 rounded-full bg-white/20 shadow-inner"></div>
                          <div className="w-2 h-2 rounded-full bg-blue-500/50 shadow-inner"></div>
                        </div>
                      </div>
                    </div>
                    <div 
                      className="absolute bottom-0 left-1/2 -translate-x-1/2 w-14 h-36 bg-gradient-to-r from-surface-lighter via-surface-dark to-surface-lighter shadow-lg flex justify-center items-end pb-6 border-x border-white/5"
                      style={{clipPath: 'polygon(0 0, 100% 0, 80% 100%, 20% 100%)'}}
                    >
                      <div className="w-8 h-10 bg-surface-dark rounded border border-white/10 shadow-[inset_0_2px_4px_rgba(0,0,0,0.5)]"></div>
                    </div>
                  </div>
                  <div className="absolute inset-0 rounded-full border-[24px] border-surface-lighter shadow-[inset_0_4px_6px_rgba(0,0,0,0.6),inset_0_-2px_4px_rgba(255,255,255,0.1),0_0_20px_rgba(0,0,0,0.5)] z-20"></div>
                  <div className="absolute inset-[2px] rounded-full border border-dashed border-white/10 opacity-50 z-20 pointer-events-none"></div>
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 size-24 bg-gradient-to-br from-surface-lighter to-surface-dark rounded-full shadow-[0_4px_15px_rgba(0,0,0,0.6),inset_0_1px_2px_rgba(255,255,255,0.2)] z-30 flex items-center justify-center border border-white/10">
                    <div className="size-16 rounded-full bg-surface-dark shadow-[inset_0_2px_5px_rgba(0,0,0,0.8)] flex items-center justify-center border border-white/5">
                      <span className="material-symbols-outlined text-gray-500 text-3xl drop-shadow-lg">directions_car</span>
                    </div>
                  </div>
                  <div className="absolute top-[-10px] left-1/2 -translate-x-1/2 w-4 h-8 bg-purple-500 rounded-sm shadow-[0_0_10px_rgba(168,85,247,0.8)] z-40 border border-white/20"></div>
                </div>
              </div>
              <div className="w-full max-w-[280px] mt-8 flex justify-between items-center text-xs font-mono text-gray-500 font-medium">
                <span className="text-left w-12">-90°</span>
                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-600 to-transparent mx-2 relative">
                  <div className="absolute left-1/2 -translate-x-1/2 -top-1 w-px h-3 bg-gray-400"></div>
                </div>
                <span className="text-right w-12">+90°</span>
              </div>
            </div>
          </div>

          {/* Brake and Throttle */}
          <div className="lg:col-span-3 grid grid-rows-2 gap-6 h-full">
            {/* Brake */}
            <div className="glass-panel rounded-2xl p-5 flex flex-col relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-accent-red/10 rounded-full blur-[40px] -mr-10 -mt-10 pointer-events-none"></div>
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                  <div className="size-8 rounded-lg bg-accent-red/10 flex items-center justify-center text-accent-red border border-accent-red/20">
                    <span className="material-symbols-outlined text-lg">arrow_downward</span>
                  </div>
                  <span className="text-sm font-bold font-display uppercase tracking-wider text-gray-300">Brake</span>
                </div>
                <span className="text-2xl font-bold font-mono text-accent-red tabular-nums">{Math.round(brake)}%</span>
              </div>
              <div className="flex-1 flex items-center gap-4">
                <div className="h-full w-full bg-surface-dark rounded-xl border border-white/5 relative p-1 shadow-inner flex flex-col justify-end overflow-hidden">
                  <div className="absolute inset-0 z-0 flex flex-col justify-between py-2 px-2 opacity-20">
                    {[...Array(5)].map((_, i) => (
                      <div key={i} className="w-full h-px bg-white"></div>
                    ))}
                  </div>
                  <div 
                    className="w-full bg-gradient-to-t from-red-900 to-accent-red rounded-lg relative transition-all duration-200 glow-red z-10 group"
                    style={{height: `${brake}%`}}
                  >
                    <div className="absolute top-0 left-0 w-full h-[1px] bg-white/50 shadow-[0_0_10px_white]"></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Throttle */}
            <div className="glass-panel rounded-2xl p-5 flex flex-col relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-accent-green/10 rounded-full blur-[40px] -mr-10 -mt-10 pointer-events-none"></div>
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                  <div className="size-8 rounded-lg bg-accent-green/10 flex items-center justify-center text-accent-green border border-accent-green/20">
                    <span className="material-symbols-outlined text-lg">arrow_upward</span>
                  </div>
                  <span className="text-sm font-bold font-display uppercase tracking-wider text-gray-300">Accelerate</span>
                </div>
                <span className="text-2xl font-bold font-mono text-accent-green tabular-nums">{Math.round(throttle)}%</span>
              </div>
              <div className="flex-1 flex items-center gap-4">
                <div className="h-full w-full bg-surface-dark rounded-xl border border-white/5 relative p-1 shadow-inner flex flex-col justify-end overflow-hidden">
                  <div className="absolute inset-0 z-0 flex flex-col justify-between py-2 px-2 opacity-20">
                    {[...Array(5)].map((_, i) => (
                      <div key={i} className="w-full h-px bg-white"></div>
                    ))}
                  </div>
                  <div 
                    className="w-full bg-gradient-to-t from-green-900 to-accent-green rounded-lg relative transition-all duration-200 glow-green z-10"
                    style={{height: `${throttle}%`}}
                  >
                    <div className="absolute top-0 left-0 w-full h-[1px] bg-white/50 shadow-[0_0_10px_white]"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
