import React, { useState, useEffect } from 'react';

interface LoaderProps {
  text?: string;
}

export function Loader({ text }: LoaderProps = {}) {
  const [cycle, setCycle] = useState(0);
  const [visibleCount, setVisibleCount] = useState(0);

  useEffect(() => {
    // Reset and start new cycle
    setVisibleCount(0);

    // Reveal one amud at a time
    const revealInterval = setInterval(() => {
      setVisibleCount(v => {
        if (v >= 9) {
          clearInterval(revealInterval);
          return v;
        }
        return v + 1;
      });
    }, 1350); // ~1.35s per amud = ~12s total

    // Full cycle reset
    const cycleTimeout = setTimeout(() => {
      setCycle(c => c + 1);
    }, 14000);

    return () => {
      clearInterval(revealInterval);
      clearTimeout(cycleTimeout);
    };
  }, [cycle]);

  const pageWidth = 52;
  const pageHeight = 68;
  const gap = 3;

  // Grid positions - tighter spacing
  const gridPositions = [
    { x: 0, y: 0 },                                    // 0: Center
    { x: 0, y: -(pageHeight + gap) },                  // 1: Top
    { x: (pageWidth + gap), y: 0 },                    // 2: Right
    { x: 0, y: (pageHeight + gap) },                   // 3: Bottom
    { x: -(pageWidth + gap), y: 0 },                   // 4: Left
    { x: (pageWidth + gap), y: -(pageHeight + gap) },  // 5: Top-right
    { x: (pageWidth + gap), y: (pageHeight + gap) },   // 6: Bottom-right
    { x: -(pageWidth + gap), y: (pageHeight + gap) },  // 7: Bottom-left
    { x: -(pageWidth + gap), y: -(pageHeight + gap) }, // 8: Top-left
  ];

  // ═══════════════════════════════════════════════════════════
  // GEMARA LAYOUT
  // ═══════════════════════════════════════════════════════════

  const GemaraAmud = ({ isAnimating }: { isAnimating: boolean }) => {
    const m = 2.5;
    const rashiW = 12;
    const tosafotW = 12;
    const expandPoint = pageHeight * 0.68;

    return (
      <g>
        {/* Rashi - Right column, full height */}
        {Array.from({ length: 14 }, (_, i) => (
          <rect
            key={`rashi-${i}`}
            x={pageWidth/2 - m - rashiW}
            y={-pageHeight/2 + m + i * 4.5}
            width={rashiW - (i % 3 === 2 ? 2 : 0)}
            height={1.5}
            rx={0.5}
            fill="#4A6C6F"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.3 + i * 0.04}s` : '0s'
            }}
          />
        ))}

        {/* Gemara upper section */}
        {Array.from({ length: 8 }, (_, i) => (
          <rect
            key={`gemara-upper-${i}`}
            x={-pageWidth/2 + m + tosafotW + 3}
            y={-pageHeight/2 + m + 1 + i * 4.5}
            width={pageWidth - rashiW - tosafotW - m * 2 - 6 - (i % 3 === 2 ? 3 : 0)}
            height={2}
            rx={0.5}
            fill="#2C2825"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.15 + i * 0.05}s` : '0s'
            }}
          />
        ))}

        {/* Gemara lower expanded section */}
        {Array.from({ length: 4 }, (_, i) => (
          <rect
            key={`gemara-lower-${i}`}
            x={-pageWidth/2 + m + 1}
            y={-pageHeight/2 + expandPoint + 2 + i * 4.5}
            width={pageWidth - rashiW - m * 2 - 4 - (i % 2 === 1 ? 4 : 0)}
            height={2}
            rx={0.5}
            fill="#2C2825"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.6 + i * 0.05}s` : '0s'
            }}
          />
        ))}

        {/* Tosafot - Left column, ends early */}
        {Array.from({ length: 8 }, (_, i) => (
          <rect
            key={`tosafot-${i}`}
            x={-pageWidth/2 + m}
            y={-pageHeight/2 + m + i * 4.5}
            width={tosafotW - (i % 3 === 1 ? 2 : 0)}
            height={1.5}
            rx={0.5}
            fill="#4A6C6F"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.4 + i * 0.04}s` : '0s'
            }}
          />
        ))}
      </g>
    );
  };

  // ═══════════════════════════════════════════════════════════
  // SHULCHAN ARUCH LAYOUT
  // ═══════════════════════════════════════════════════════════

  const ShulchanAruchAmud = ({ isAnimating }: { isAnimating: boolean }) => {
    const m = 2.5;
    const colW = (pageWidth - m * 2 - 4) / 3;
    const topH = pageHeight * 0.42;

    return (
      <g>
        {/* Top 3 columns */}
        {[0, 1, 2].map(col => (
          Array.from({ length: 5 }, (_, i) => (
            <rect
              key={`top-${col}-${i}`}
              x={-pageWidth/2 + m + col * (colW + 2)}
              y={-pageHeight/2 + m + i * 5}
              width={colW - (i % 2 === 1 ? 2 : 0)}
              height={col === 1 ? 2 : 1.5}
              rx={0.5}
              fill={col === 1 ? '#2C2825' : '#4A6C6F'}
              style={{
                opacity: isAnimating ? 0 : 0.85,
                animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
                animationDelay: isAnimating ? `${0.1 + col * 0.12 + i * 0.04}s` : '0s'
              }}
            />
          ))
        ))}

        {/* Bottom varying section */}
        {Array.from({ length: 7 }, (_, i) => {
          const isTwoCol = i < 3;
          const yPos = -pageHeight/2 + topH + 4 + i * 4;

          if (isTwoCol) {
            return [0, 1].map(col => (
              <rect
                key={`btm-${i}-${col}`}
                x={-pageWidth/2 + m + col * (pageWidth/2 - m)}
                y={yPos}
                width={(pageWidth - m * 2) / 2 - 2 - (i % 2 === 0 ? 2 : 0)}
                height={1.5}
                rx={0.5}
                fill="#2C2825"
                style={{
                  opacity: isAnimating ? 0 : 0.85,
                  animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
                  animationDelay: isAnimating ? `${0.5 + i * 0.05}s` : '0s'
                }}
              />
            ));
          }
          return (
            <rect
              key={`btm-${i}`}
              x={-pageWidth/2 + m}
              y={yPos}
              width={pageWidth - m * 2 - (i % 2 === 1 ? 6 : 0)}
              height={1.5}
              rx={0.5}
              fill="#4A6C6F"
              style={{
                opacity: isAnimating ? 0 : 0.85,
                animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
                animationDelay: isAnimating ? `${0.6 + i * 0.04}s` : '0s'
              }}
            />
          );
        })}
      </g>
    );
  };

  // ═══════════════════════════════════════════════════════════
  // MIKRAOT GEDOLOT LAYOUT
  // ═══════════════════════════════════════════════════════════

  const MikraotGedolotAmud = ({ isAnimating }: { isAnimating: boolean }) => {
    const m = 2.5;
    const centerW = 22;
    const centerH = 32;
    const sideW = 10;

    return (
      <g>
        {/* Center block */}
        {Array.from({ length: 6 }, (_, i) => (
          <rect
            key={`center-${i}`}
            x={-centerW/2 + 2}
            y={-centerH/2 - 6 + i * 5}
            width={centerW - 4 - (i % 2 === 1 ? 3 : 0)}
            height={2}
            rx={0.5}
            fill="#2C2825"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.1 + i * 0.05}s` : '0s'
            }}
          />
        ))}

        {/* Left commentary */}
        {Array.from({ length: 10 }, (_, i) => (
          <rect
            key={`left-${i}`}
            x={-pageWidth/2 + m}
            y={-pageHeight/2 + m + i * 4}
            width={sideW - (i % 2 === 0 ? 2 : 0)}
            height={1.2}
            rx={0.5}
            fill="#4A6C6F"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.35 + i * 0.03}s` : '0s'
            }}
          />
        ))}

        {/* Right commentary */}
        {Array.from({ length: 10 }, (_, i) => (
          <rect
            key={`right-${i}`}
            x={pageWidth/2 - m - sideW + (i % 2 === 1 ? 2 : 0)}
            y={-pageHeight/2 + m + i * 4}
            width={sideW - (i % 2 === 1 ? 2 : 0)}
            height={1.2}
            rx={0.5}
            fill="#4A6C6F"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.4 + i * 0.03}s` : '0s'
            }}
          />
        ))}

        {/* Bottom commentary */}
        {Array.from({ length: 5 }, (_, i) => (
          <rect
            key={`btm-${i}`}
            x={-pageWidth/2 + m + 4}
            y={pageHeight/2 - m - 20 + i * 4}
            width={pageWidth - m * 2 - 8 - (i % 2 === 0 ? 8 : 0)}
            height={1.2}
            rx={0.5}
            fill="#8B7355"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.55 + i * 0.04}s` : '0s'
            }}
          />
        ))}
      </g>
    );
  };

  // ═══════════════════════════════════════════════════════════
  // MISHNAH LAYOUT
  // ═══════════════════════════════════════════════════════════

  const MishnahAmud = ({ isAnimating }: { isAnimating: boolean }) => {
    const m = 2.5;
    const colW = (pageWidth - m * 2 - 3) / 2;
    const mainH = pageHeight * 0.62;

    return (
      <g>
        {/* Two columns */}
        {[0, 1].map(col => (
          Array.from({ length: 8 }, (_, i) => (
            <rect
              key={`main-${col}-${i}`}
              x={-pageWidth/2 + m + col * (colW + 3)}
              y={-pageHeight/2 + m + i * 5}
              width={colW - (i % 3 === 2 ? 3 : 0)}
              height={2}
              rx={0.5}
              fill="#2C2825"
              style={{
                opacity: isAnimating ? 0 : 0.85,
                animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
                animationDelay: isAnimating ? `${0.1 + col * 0.15 + i * 0.04}s` : '0s'
              }}
            />
          ))
        ))}

        {/* Bottom commentary */}
        {Array.from({ length: 5 }, (_, i) => (
          <rect
            key={`cmnt-${i}`}
            x={-pageWidth/2 + m}
            y={-pageHeight/2 + mainH + 4 + i * 4.5}
            width={pageWidth - m * 2 - (i % 2 === 1 ? 10 : i % 3 === 0 ? 4 : 0)}
            height={1.5}
            rx={0.5}
            fill="#4A6C6F"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.55 + i * 0.04}s` : '0s'
            }}
          />
        ))}
      </g>
    );
  };

  // ═══════════════════════════════════════════════════════════
  // RAMBAM LAYOUT
  // ═══════════════════════════════════════════════════════════

  const RambamAmud = ({ isAnimating }: { isAnimating: boolean }) => {
    const m = 2.5;

    return (
      <g>
        {/* Header */}
        <rect
          x={-pageWidth/4}
          y={-pageHeight/2 + m}
          width={pageWidth/2}
          height={2.5}
          rx={0.5}
          fill="#D97757"
          style={{
            opacity: isAnimating ? 0 : 0.85,
            animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
            animationDelay: isAnimating ? '0.1s' : '0s'
          }}
        />

        {/* Main text */}
        {Array.from({ length: 13 }, (_, i) => (
          <rect
            key={`text-${i}`}
            x={-pageWidth/2 + m}
            y={-pageHeight/2 + m + 8 + i * 4.5}
            width={pageWidth - m * 2 - (i % 4 === 3 ? 12 : i % 2 === 1 ? 4 : 0)}
            height={2}
            rx={0.5}
            fill="#2C2825"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.2 + i * 0.04}s` : '0s'
            }}
          />
        ))}
      </g>
    );
  };

  // ═══════════════════════════════════════════════════════════
  // TUR LAYOUT - 2 columns with header
  // ═══════════════════════════════════════════════════════════

  const TurAmud = ({ isAnimating }: { isAnimating: boolean }) => {
    const m = 2.5;
    const colW = (pageWidth - m * 2 - 4) / 2;

    return (
      <g>
        {/* Header spanning both columns */}
        <rect
          x={-pageWidth/2 + m + 8}
          y={-pageHeight/2 + m}
          width={pageWidth - m * 2 - 16}
          height={2.5}
          rx={0.5}
          fill="#D97757"
          style={{
            opacity: isAnimating ? 0 : 0.85,
            animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
            animationDelay: isAnimating ? '0.1s' : '0s'
          }}
        />

        {/* Two columns */}
        {[0, 1].map(col => (
          Array.from({ length: 12 }, (_, i) => (
            <rect
              key={`col-${col}-${i}`}
              x={-pageWidth/2 + m + col * (colW + 4)}
              y={-pageHeight/2 + m + 8 + i * 4.5}
              width={colW - (i % 3 === 1 ? 3 : 0)}
              height={1.8}
              rx={0.5}
              fill={col === 0 ? '#2C2825' : '#4A6C6F'}
              style={{
                opacity: isAnimating ? 0 : 0.85,
                animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
                animationDelay: isAnimating ? `${0.15 + col * 0.2 + i * 0.035}s` : '0s'
              }}
            />
          ))
        ))}
      </g>
    );
  };

  // ═══════════════════════════════════════════════════════════
  // SIDDUR LAYOUT - Centered text blocks
  // ═══════════════════════════════════════════════════════════

  const SiddurAmud = ({ isAnimating }: { isAnimating: boolean }) => {
    const m = 2.5;

    return (
      <g>
        {/* Centered lines with varying widths */}
        {Array.from({ length: 14 }, (_, i) => {
          const widthVariation = i % 4 === 0 ? 0.5 : i % 3 === 0 ? 0.7 : i % 2 === 0 ? 0.85 : 1;
          const lineWidth = (pageWidth - m * 2 - 8) * widthVariation;

          return (
            <rect
              key={`line-${i}`}
              x={-lineWidth / 2}
              y={-pageHeight/2 + m + i * 4.5}
              width={lineWidth}
              height={2}
              rx={0.5}
              fill={i % 5 === 0 ? '#D97757' : '#2C2825'}
              style={{
                opacity: isAnimating ? 0 : 0.85,
                animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
                animationDelay: isAnimating ? `${0.1 + i * 0.045}s` : '0s'
              }}
            />
          );
        })}
      </g>
    );
  };

  // ═══════════════════════════════════════════════════════════
  // ZOHAR LAYOUT - Narrow center column with marginal notes
  // ═══════════════════════════════════════════════════════════

  const ZoharAmud = ({ isAnimating }: { isAnimating: boolean }) => {
    const m = 2.5;
    const centerW = 28;
    const marginW = 8;

    return (
      <g>
        {/* Main text - narrow center */}
        {Array.from({ length: 14 }, (_, i) => (
          <rect
            key={`main-${i}`}
            x={-centerW/2}
            y={-pageHeight/2 + m + i * 4.5}
            width={centerW - (i % 3 === 2 ? 4 : 0)}
            height={2}
            rx={0.5}
            fill="#2C2825"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.1 + i * 0.04}s` : '0s'
            }}
          />
        ))}

        {/* Left margin notes */}
        {Array.from({ length: 6 }, (_, i) => (
          <rect
            key={`left-${i}`}
            x={-pageWidth/2 + m}
            y={-pageHeight/2 + m + 8 + i * 9}
            width={marginW}
            height={1}
            rx={0.5}
            fill="#8B7355"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.4 + i * 0.05}s` : '0s'
            }}
          />
        ))}

        {/* Right margin notes */}
        {Array.from({ length: 6 }, (_, i) => (
          <rect
            key={`right-${i}`}
            x={pageWidth/2 - m - marginW}
            y={-pageHeight/2 + m + 12 + i * 9}
            width={marginW}
            height={1}
            rx={0.5}
            fill="#8B7355"
            style={{
              opacity: isAnimating ? 0 : 0.85,
              animation: isAnimating ? 'lineAppear 0.2s ease-out forwards' : 'none',
              animationDelay: isAnimating ? `${0.45 + i * 0.05}s` : '0s'
            }}
          />
        ))}
      </g>
    );
  };

  // Assign layouts to positions
  const amudLayouts = [
    GemaraAmud,           // 0: Center
    ShulchanAruchAmud,    // 1: Top
    MikraotGedolotAmud,   // 2: Right
    MishnahAmud,          // 3: Bottom
    RambamAmud,           // 4: Left
    TurAmud,              // 5: Top-right
    SiddurAmud,           // 6: Bottom-right
    ZoharAmud,            // 7: Bottom-left
    GemaraAmud,           // 8: Top-left
  ];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '60px 20px',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      marginBottom: '24px'
    }}>
      <div style={{ position: 'relative', width: '280px', height: '300px' }} key={cycle}>
        <svg viewBox="-140 -150 280 300" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
          <defs>
            <filter id="softGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="1" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Amudim - one at a time */}
          {gridPositions.map((pos, i) => {
            if (i >= visibleCount) return null;

            const AmudComponent = amudLayouts[i];
            // Only animate the most recently appeared amud
            const isAnimating = i === visibleCount - 1;

            return (
              <g
                key={`${cycle}-${i}`}
                style={{
                  animation: isAnimating ? 'slideOut 0.5s cubic-bezier(0.34, 1.3, 0.64, 1) forwards' : 'none',
                  opacity: isAnimating ? undefined : 0.85,
                  '--target-x': `${pos.x}px`,
                  '--target-y': `${pos.y}px`,
                } as React.CSSProperties}
              >
                <g transform={`translate(${pos.x}, ${pos.y})`}>
                  {/* Page background */}
                  <rect
                    x={-pageWidth/2}
                    y={-pageHeight/2}
                    width={pageWidth}
                    height={pageHeight}
                    rx={1.5}
                    fill="#FAF7F2"
                    stroke="#D97757"
                    strokeWidth={0.8}
                    filter="url(#softGlow)"
                  />

                  {/* Amud content */}
                  <AmudComponent isAnimating={isAnimating} />
                </g>
              </g>
            );
          })}

          {/* Pulse when new amud appears */}
          {visibleCount > 0 && visibleCount <= 9 && (
            <circle
              key={`pulse-${visibleCount}`}
              cx="0"
              cy="0"
              r="10"
              fill="none"
              stroke="#D97757"
              strokeWidth="0.8"
              style={{
                animation: 'pulse 1s ease-out forwards'
              }}
            />
          )}
        </svg>
      </div>

      {/* Status */}
      <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
        <p style={{
          fontSize: '12px',
          fontWeight: '500',
          letterSpacing: '2.5px',
          textTransform: 'uppercase',
          color: '#4A6C6F',
          margin: 0,
          animation: 'textPulse 2s ease-in-out infinite'
        }}>
          {text || 'Searching...'}
        </p>
      </div>

      <style>{`
        @keyframes slideOut {
          0% { opacity: 0; transform: scaleX(0); transform-origin: left; }
          100% { opacity: 0.85; transform: scaleX(1); }
        }
        @keyframes lineAppear {
          0% { opacity: 0; transform: scaleX(0); transform-origin: left; }
          100% { opacity: 0.85; transform: scaleX(1); }
        }

        @keyframes pulse {
          0% {
            r: 10;
            opacity: 0.6;
          }
          100% {
            r: 100;
            opacity: 0;
          }
        }

        @keyframes textPulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
