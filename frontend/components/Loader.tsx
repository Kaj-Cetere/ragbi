"use client";

import { useEffect, useState } from "react";
import styles from "./Loader.module.css";

interface LoaderProps {
  text?: string;
}

const msgs = [
  { t: "Expanding Knowledge...", d: 0 },
  { t: "Illuminating Connections...", d: 2500 },
  { t: "Full Synthesis...", d: 6000 },
  { t: "Converging...", d: 9000 }
];

export function Loader({ text }: LoaderProps) {
  const [currentText, setCurrentText] = useState(text || msgs[0].t);

  useEffect(() => {
    // If text prop is provided, override the animation text
    if (text) {
      setCurrentText(text);
      return;
    }

    let timeouts: NodeJS.Timeout[] = [];
    let interval: NodeJS.Timeout;

    // Cycle the text
    const cycleText = () => {
        msgs.forEach((item) => {
          const timeout = setTimeout(() => {
            setCurrentText(item.t);
          }, item.d);
          timeouts.push(timeout);
        });
    }

    cycleText();
    // Restart cycle every 12s to match animation
    interval = setInterval(() => {
        cycleText();
    }, 12000);

    return () => {
      timeouts.forEach(clearTimeout);
      clearInterval(interval);
    };
  }, [text]);

  const renderGen3 = () => {
    const angles = [-25, 25];
    return angles.map((ang, i) => (
      <div
        key={`g3-${i}`}
        className={`${styles.limbWrapper} ${styles.g3Wrapper}`}
        style={{ "--r": `${ang}deg` } as React.CSSProperties}
      >
        <div className={`${styles.extender} ${styles.g3Extender}`}>
          <div className={styles.line}></div>
          <div className={`${styles.orb} ${styles.g3Orb}`}></div>
        </div>
      </div>
    ));
  };

  const renderGen2 = () => {
    const angles = [-40, 0, 40];
    return angles.map((ang, i) => (
      <div
        key={`g2-${i}`}
        className={`${styles.limbWrapper} ${styles.g2Wrapper}`}
        style={{ "--r": `${ang}deg` } as React.CSSProperties}
      >
        <div className={`${styles.extender} ${styles.g2Extender}`}>
          <div className={styles.line}></div>
          <div className={`${styles.orb} ${styles.g2Orb}`}></div>
          {renderGen3()}
        </div>
      </div>
    ));
  };

  const renderGen1 = () => {
    const arms = Array.from({ length: 6 });
    return arms.map((_, i) => {
      const r1 = i * 60;
      return (
        <div
          key={`g1-${i}`}
          className={styles.limbWrapper}
          style={{ "--r": `${r1}deg` } as React.CSSProperties}
        >
          <div className={`${styles.extender} ${styles.g1Extender}`}>
            <div className={styles.line}></div>
            <div className={`${styles.orb} ${styles.g1Orb}`}></div>
            {renderGen2()}
          </div>
        </div>
      );
    });
  };

  return (
    <div className="flex flex-col items-center justify-center w-full overflow-hidden rounded-xl py-10 my-8">
        <div className={styles.loaderWrapper}>
            <div className={styles.constellationSystem}>
                {/* Seed */}
                <div className={`${styles.orb} ${styles.seed}`}></div>
                {/* Tree */}
                {renderGen1()}
            </div>
        </div>

        <div className={styles.statusContainer}>
            <div className={styles.statusText}>{currentText}</div>
        </div>
    </div>
  );
}
