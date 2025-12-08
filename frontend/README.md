# Kesher AI Frontend

A beautiful, modern React frontend for the Sefaria RAG application.

## Features

- **Bioluminescent Theme**: Deep forest/charcoal tones with electric lime accents
- **Streaming Responses**: Real-time AI responses with smooth animations
- **Citation Cards**: Interactive source citations with Hebrew text
- **Sidebar Reader**: View full chapter context from Sefaria API
- **Mobile Responsive**: Works beautifully on all screen sizes

## Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Lucide Icons** - Beautiful icon set

## Getting Started

### Prerequisites

- Node.js 18+ 
- The FastAPI backend running on port 8000

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/
│   ├── globals.css      # Global styles & Tailwind config
│   ├── layout.tsx       # Root layout with fonts
│   └── page.tsx         # Main chat interface
├── components/
│   ├── CitationCard.tsx # Source citation cards
│   ├── Loader.tsx       # Organic loading animation
│   ├── Sidebar.tsx      # Sefaria text reader sidebar
│   └── SourcesList.tsx  # Additional sources list
├── utils/
│   ├── api.ts           # API utilities & types
│   └── cn.ts            # Class name utilities
└── tailwind.config.ts   # Theme configuration
```

## API Endpoints

The frontend expects these endpoints from the FastAPI backend:

- `POST /api/chat/stream` - Stream chat responses (SSE)
- `GET /api/sefaria/text/{ref}` - Fetch Sefaria text for sidebar

## Customization

### Theme Colors

Edit `tailwind.config.ts` to customize the color palette:

```typescript
colors: {
  bg: {
    deep: "#050f0a",      // Main background
    surface: "#0f1f18",   // Cards/surfaces
    elevated: "#162921",  // Hover states
  },
  accent: {
    primary: "#4ade80",   // Electric Lime
    secondary: "#c084fc", // Electric Lavender
    tertiary: "#2dd4bf",  // Teal
  }
}
```

### Fonts

The app uses three font families:
- **Manrope** - Modern sans-serif for UI
- **Crimson Text** - Elegant serif for reading
- **Frank Ruhl Libre** - Hebrew text display
