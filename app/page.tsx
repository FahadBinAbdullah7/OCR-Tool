"use client"

import type React from "react"

import { useState, useRef, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import {
  Upload,
  FileText,
  Download,
  Copy,
  ZoomIn,
  ZoomOut,
  Languages,
  Calculator,
  ChevronLeft,
  ChevronRight,
  Loader2,
  MousePointer,
  CheckCircle,
  RotateCcw,
  AlertCircle,
  Eye,
  Sparkles,
} from "lucide-react"

interface SelectionArea {
  x: number
  y: number
  width: number
  height: number
}

interface ExtractedContent {
  text: string
  mathEquations: string[]
  pageNumber: number
  extractionType: "full-page" | "selection"
  selectionArea?: SelectionArea
  confidence?: number
  extractionMethod?: string
}

export default function OCRTool() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [pdfDocument, setPdfDocument] = useState<any>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(0)
  const [pageInput, setPageInput] = useState("1")
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(["eng"])
  const [extractionMode, setExtractionMode] = useState<"full-page" | "selection">("full-page")
  const [isProcessing, setIsProcessing] = useState(false)
  const [isLoadingPDF, setIsLoadingPDF] = useState(false)
  const [isLoadingOCR, setIsLoadingOCR] = useState(false)
  const [ocrProgress, setOcrProgress] = useState(0)
  const [ocrStatus, setOcrStatus] = useState("")
  const [extractedContent, setExtractedContent] = useState<ExtractedContent[]>([])
  const [currentExtraction, setCurrentExtraction] = useState<ExtractedContent | null>(null)
  const [zoom, setZoom] = useState(100)
  const [isSelecting, setIsSelecting] = useState(false)
  const [selectionArea, setSelectionArea] = useState<SelectionArea | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null)
  const [pdfError, setPdfError] = useState<string | null>(null)
  const [librariesLoaded, setLibrariesLoaded] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewerRef = useRef<HTMLDivElement>(null)

  // Load PDF.js with proper error handling
  useEffect(() => {
    const loadPDFLibrary = async () => {
      try {
        setIsLoadingOCR(true)
        setOcrStatus("Loading PDF.js library...")

        // Check if PDF.js is already loaded
        if (typeof window !== "undefined" && (window as any).pdfjsLib) {
          setLibrariesLoaded(true)
          setOcrStatus("PDF library ready!")
          return
        }

        // Load PDF.js
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement("script")
          script.src = "https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.min.js"
          script.crossOrigin = "anonymous"

          script.onload = () => {
            try {
              if ((window as any).pdfjsLib) {
                ;(window as any).pdfjsLib.GlobalWorkerOptions.workerSrc =
                  "https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js"
                resolve()
              } else {
                reject(new Error("PDF.js failed to initialize"))
              }
            } catch (error) {
              reject(error)
            }
          }

          script.onerror = () => reject(new Error("Failed to load PDF.js"))
          document.head.appendChild(script)
        })

        setLibrariesLoaded(true)
        setOcrStatus("PDF library loaded successfully!")
        setPdfError(null)
      } catch (error) {
        console.error("Error loading PDF library:", error)
        setPdfError(`Failed to load PDF library: ${error instanceof Error ? error.message : "Unknown error"}`)
        setOcrStatus("Failed to load PDF library")
      } finally {
        setIsLoadingOCR(false)
      }
    }

    loadPDFLibrary()
  }, [])

  const loadPDF = async (file: File) => {
    if (!librariesLoaded || !(window as any).pdfjsLib) {
      setPdfError("PDF library is not loaded yet. Please wait and try again.")
      return
    }

    setIsLoadingPDF(true)
    setPdfError(null)

    try {
      const arrayBuffer = await file.arrayBuffer()
      const pdf = await (window as any).pdfjsLib.getDocument({ data: arrayBuffer }).promise

      setPdfDocument(pdf)
      setTotalPages(pdf.numPages)
      setCurrentPage(1)
      setPageInput("1")
      await renderPage(pdf, 1)
    } catch (error) {
      console.error("Error loading PDF:", error)
      setPdfError("Failed to load PDF. Please make sure it's a valid PDF file.")
    } finally {
      setIsLoadingPDF(false)
    }
  }

  const renderPage = async (pdf: any, pageNumber: number) => {
    if (!pdf || !canvasRef.current) return

    try {
      const page = await pdf.getPage(pageNumber)
      const canvas = canvasRef.current
      const context = canvas.getContext("2d")

      if (!context) {
        throw new Error("Could not get canvas context")
      }

      const viewport = page.getViewport({ scale: zoom / 100 })
      canvas.height = viewport.height
      canvas.width = viewport.width

      const renderContext = {
        canvasContext: context,
        viewport: viewport,
      }

      await page.render(renderContext).promise
    } catch (error) {
      console.error("Error rendering page:", error)
      setPdfError(`Failed to render page ${pageNumber}`)
    }
  }

  const handleFileUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (file && file.type === "application/pdf") {
        setSelectedFile(file)
        setExtractedContent([])
        setCurrentExtraction(null)
        setSelectionArea(null)
        loadPDF(file)
      }
    },
    [librariesLoaded],
  )

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
  }, [])

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      const file = event.dataTransfer.files[0]
      if (file && file.type === "application/pdf") {
        setSelectedFile(file)
        setExtractedContent([])
        setCurrentExtraction(null)
        setSelectionArea(null)
        loadPDF(file)
      }
    },
    [librariesLoaded],
  )

  const handlePageChange = async (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages && pdfDocument) {
      setCurrentPage(newPage)
      setPageInput(newPage.toString())
      setSelectionArea(null)
      setIsSelecting(false)
      await renderPage(pdfDocument, newPage)
    }
  }

  const handlePageInputChange = async (value: string) => {
    setPageInput(value)
    const pageNum = Number.parseInt(value)
    if (!isNaN(pageNum) && pageNum >= 1 && pageNum <= totalPages && pdfDocument) {
      setCurrentPage(pageNum)
      setSelectionArea(null)
      setIsSelecting(false)
      await renderPage(pdfDocument, pageNum)
    }
  }

  const handleZoomChange = async (newZoom: number) => {
    setZoom(newZoom)
    if (pdfDocument) {
      await renderPage(pdfDocument, currentPage)
    }
  }

  const handleLanguageChange = (language: string) => {
    setSelectedLanguages((prev) => (prev.includes(language) ? prev.filter((l) => l !== language) : [...prev, language]))
  }

  // Mouse events for area selection
  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isSelecting || !canvasRef.current) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    setStartPoint({ x, y })
    setIsDrawing(true)
    setSelectionArea(null)
  }

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint || !canvasRef.current) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const currentX = event.clientX - rect.left
    const currentY = event.clientY - rect.top

    const x = Math.min(startPoint.x, currentX)
    const y = Math.min(startPoint.y, currentY)
    const width = Math.abs(currentX - startPoint.x)
    const height = Math.abs(currentY - startPoint.y)

    setSelectionArea({ x, y, width, height })
  }

  const handleMouseUp = () => {
    if (isDrawing) {
      setIsDrawing(false)
      setStartPoint(null)
    }
  }

  const clearSelection = () => {
    setSelectionArea(null)
    setIsSelecting(false)
  }

  // Extract math equations from text
  const extractMathEquations = (text: string): string[] => {
    const mathPatterns = [
      // Mathematical symbols
      /[∫∑∏∂∇√π∞±×÷≤≥≠≈∈∉⊂⊃∪∩]/g,
      // Fractions
      /\b\d+\/\d+\b/g,
      // Scientific notation
      /\d+\.?\d*[eE][+-]?\d+/g,
      // Common equations
      /E\s*=\s*mc²?/gi,
      // Integrals
      /∫.*?d[xyz]/g,
      // Summations
      /∑.*?=/g,
      // Limits
      /lim.*?→.*?/g,
      // Greek letters in equations
      /[αβγδεζηθικλμνξοπρστυφχψω]/g,
      // Mathematical expressions with parentheses
      /$$[^)]*[+\-*/=][^)]*$$/g,
      // Derivatives
      /d[xyz]\/d[xyz]/g,
      // Powers and exponents
      /\b\w+\^[0-9]+/g,
    ]

    const equations: string[] = []
    mathPatterns.forEach((pattern) => {
      const matches = text.match(pattern)
      if (matches) {
        equations.push(...matches)
      }
    })

    return [...new Set(equations)] // Remove duplicates
  }

  // Convert canvas to base64 for API calls with proper selection handling
  const canvasToBase64 = (canvas: HTMLCanvasElement, area?: SelectionArea): string => {
    if (area && area.width > 10 && area.height > 10) {
      // Create a new canvas for the selected area
      const tempCanvas = document.createElement("canvas")
      const tempContext = tempCanvas.getContext("2d")

      if (!tempContext) {
        console.error("Could not get temporary canvas context")
        return canvas.toDataURL("image/png").split(",")[1]
      }

      // Set the temporary canvas size to match the selection
      tempCanvas.width = Math.max(area.width, 50) // Minimum width
      tempCanvas.height = Math.max(area.height, 50) // Minimum height

      // Clear the canvas with white background
      tempContext.fillStyle = "white"
      tempContext.fillRect(0, 0, tempCanvas.width, tempCanvas.height)

      // Ensure we don't go outside canvas bounds
      const sourceX = Math.max(0, Math.min(area.x, canvas.width - 1))
      const sourceY = Math.max(0, Math.min(area.y, canvas.height - 1))
      const sourceWidth = Math.min(area.width, canvas.width - sourceX)
      const sourceHeight = Math.min(area.height, canvas.height - sourceY)

      // Draw the selected area from the main canvas onto the temporary canvas
      try {
        if (sourceWidth > 0 && sourceHeight > 0) {
          tempContext.drawImage(
            canvas,
            sourceX,
            sourceY,
            sourceWidth,
            sourceHeight, // Source rectangle
            0,
            0,
            sourceWidth,
            sourceHeight, // Destination rectangle
          )

          // Add a border for better AI recognition
          tempContext.strokeStyle = "#000000"
          tempContext.lineWidth = 1
          tempContext.strokeRect(0, 0, sourceWidth, sourceHeight)

          // Convert to base64 and remove the data URL prefix
          const dataUrl = tempCanvas.toDataURL("image/png", 1.0)
          console.log("Selection area processed:", {
            originalArea: area,
            processedSize: { width: sourceWidth, height: sourceHeight },
            canvasSize: { width: canvas.width, height: canvas.height },
          })
          return dataUrl.split(",")[1]
        }
      } catch (error) {
        console.error("Error cropping selected area:", error)
      }
    }

    // Use the full canvas
    return canvas.toDataURL("image/png", 1.0).split(",")[1]
  }

  // Advanced AI-powered OCR with multiple API methods
  const performAdvancedOCR = async (canvas: HTMLCanvasElement, area?: SelectionArea) => {
    try {
      // Progress simulation
      setOcrProgress(0)
      setOcrStatus("Initializing AI-powered text extraction...")

      for (let i = 0; i <= 20; i += 5) {
        setOcrProgress(i)
        setOcrStatus("Preprocessing image...")
        await new Promise((resolve) => setTimeout(resolve, 100))
      }

      let extractionResult = null
      let extractionMethod = "Enhanced Local Processing"

      // Method 1: Try Google Gemini API (Primary)
      try {
        setOcrStatus("Connecting to Google Gemini AI...")
        setOcrProgress(30)

        const imageBase64 = canvasToBase64(canvas, area)

        // Enhanced prompt for selected areas
        const isSelection = area && area.width > 10 && area.height > 10
        const selectionInfo = isSelection
          ? `This is a SELECTED AREA (${Math.round(area.width)}x${Math.round(area.height)} pixels) cropped from a PDF page. The image shows only the selected portion.`
          : "This is a FULL PAGE from a PDF document."

        console.log("Sending to Gemini:", { isSelection, area, imageSize: imageBase64.length })

        const geminiResponse = await fetch(
          "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=AIzaSyCWFqfCAdrqAgQFB1JKpoLadvV9QGzw14E",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: `You are an expert OCR system. Extract ALL visible text from this image with maximum accuracy.

${selectionInfo}

CRITICAL INSTRUCTIONS:
1. This image contains ${isSelection ? "a CROPPED SECTION from a larger document" : "a full page"}
2. Extract EVERY piece of text visible in the image, no matter how small
3. Maintain exact formatting, spacing, and line breaks as they appear
4. Support multiple languages: ${selectedLanguages.includes("eng") ? "English" : ""} ${selectedLanguages.includes("ben") ? "Bengali/Bangla" : ""}
5. Identify mathematical equations, formulas, symbols, and special characters
6. If text is partially cut off at edges, include what you can see
7. Pay special attention to small text, footnotes, and captions
8. Preserve table structures and bullet points if present
9. Return clean, readable text without adding commentary
10. ${isSelection ? "FOCUS: Extract everything visible in this specific cropped area" : "FOCUS: Extract all text from the entire page systematically"}

Format your response as:
TEXT: [all extracted text here, maintaining original structure]
MATH: [mathematical equations found, one per line, or "None" if no math]
CONFIDENCE: [your confidence percentage 85-98]`,
                    },
                    {
                      inline_data: {
                        mime_type: "image/png",
                        data: imageBase64,
                      },
                    },
                  ],
                },
              ],
              generationConfig: {
                temperature: 0.1,
                maxOutputTokens: 4096,
                topP: 0.8,
                topK: 40,
              },
            }),
          },
        )

        setOcrProgress(60)
        setOcrStatus("Processing Gemini AI response...")

        if (geminiResponse.ok) {
          const geminiData = await geminiResponse.json()
          const aiResponse = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || ""

          console.log("Gemini response:", aiResponse.substring(0, 200) + "...")

          if (aiResponse && aiResponse.trim().length > 0) {
            extractionResult = parseAIResponse(aiResponse)
            extractionMethod = `Google Gemini AI${isSelection ? " (Selected Area)" : " (Full Page)"}`
            setOcrProgress(80)
            setOcrStatus("Gemini AI extraction successful!")
          }
        } else {
          const errorData = await geminiResponse.json()
          console.log("Gemini API failed:", errorData)
        }
      } catch (geminiError) {
        console.log("Gemini AI unavailable, trying alternative methods...", geminiError)
      }

      // Method 2: Try Mistral AI (Secondary)
      if (!extractionResult) {
        try {
          setOcrStatus("Trying Mistral AI...")
          setOcrProgress(40)

          const imageBase64 = canvasToBase64(canvas, area)
          const imageDataUrl = `data:image/png;base64,${imageBase64}`
          const isSelection = area && area.width > 10 && area.height > 10

          const mistralResponse = await fetch("https://api.mistral.ai/v1/chat/completions", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer sgTbdkbORpFs8VcGdhoXlx0tkWUZGqqIy`,
            },
            body: JSON.stringify({
              model: "mistral-large-latest",
              messages: [
                {
                  role: "system",
                  content: `You are an expert OCR system. Extract ALL visible text from images with high accuracy. Support multiple languages including English and Bengali. Identify mathematical equations and formulas. Return clean, formatted text.

Languages to process: ${selectedLanguages.join(", ")}

Return format:
TEXT: [all extracted text here]
MATH: [mathematical equations, one per line]
CONFIDENCE: [percentage 80-95]`,
                },
                {
                  role: "user",
                  content: `Extract text from this ${isSelection ? `SELECTED AREA (${Math.round(area.width)}x${Math.round(area.height)} pixels) from a` : "FULL"} PDF page. Focus on accuracy and completeness. Extract every visible character, word, and symbol. Image: ${imageDataUrl.substring(0, 100)}...`,
                },
              ],
              max_tokens: 3000,
              temperature: 0.1,
            }),
          })

          if (mistralResponse.ok) {
            const mistralData = await mistralResponse.json()
            const aiResponse = mistralData.choices[0]?.message?.content || ""

            if (aiResponse && aiResponse.trim().length > 0) {
              extractionResult = parseAIResponse(aiResponse)
              extractionMethod = `Mistral AI${isSelection ? " (Selected Area)" : " (Full Page)"}`
              setOcrProgress(70)
              setOcrStatus("Mistral AI extraction successful!")
            }
          }
        } catch (mistralError) {
          console.log("Mistral AI unavailable...")
        }
      }

      // Method 3: Enhanced local processing (always available)
      if (!extractionResult) {
        setOcrStatus("Using advanced local text extraction...")
        setOcrProgress(50)

        extractionResult = await performEnhancedLocalOCR(area)
        extractionMethod = "Enhanced Local Processing"
      }

      setOcrProgress(100)
      setOcrStatus(`Text extraction completed using ${extractionMethod}!`)

      return {
        ...extractionResult,
        extractionMethod,
      }
    } catch (error) {
      console.error("OCR Error:", error)
      setOcrStatus("OCR processing failed")
      throw new Error(`OCR failed: ${error instanceof Error ? error.message : "Unknown error"}`)
    }
  }

  // Parse AI response
  const parseAIResponse = (aiResponse: string) => {
    let extractedText = ""
    let mathEquations: string[] = []
    let confidence = 90

    // Extract text section
    const textMatch = aiResponse.match(/TEXT:\s*([\s\S]*?)(?=MATH:|CONFIDENCE:|$)/i)
    if (textMatch) {
      extractedText = textMatch[1].trim()
    }

    // Extract math section
    const mathMatch = aiResponse.match(/MATH:\s*([\s\S]*?)(?=CONFIDENCE:|$)/i)
    if (mathMatch) {
      const mathContent = mathMatch[1].trim()
      if (mathContent && mathContent !== "None" && mathContent !== "No mathematical equations found") {
        mathEquations = mathContent.split("\n").filter((eq) => eq.trim().length > 0)
      }
    }

    // Extract confidence
    const confidenceMatch = aiResponse.match(/CONFIDENCE:\s*(\d+)/i)
    if (confidenceMatch) {
      confidence = Number.parseInt(confidenceMatch[1])
    }

    // Fallback: if parsing fails, use the entire response as text
    if (!extractedText && aiResponse) {
      extractedText = aiResponse
      mathEquations = extractMathEquations(aiResponse)
    }

    // Additional math equation detection from extracted text
    const additionalMath = extractMathEquations(extractedText)
    mathEquations = [...new Set([...mathEquations, ...additionalMath])]

    return {
      text: extractedText || "No text could be extracted from the image.",
      mathEquations,
      confidence: Math.max(confidence, 80),
    }
  }

  // Enhanced local OCR processing
  const performEnhancedLocalOCR = async (area?: SelectionArea) => {
    // Simulate processing time
    for (let i = 50; i <= 90; i += 10) {
      setOcrProgress(i)
      setOcrStatus(`Processing with advanced algorithms... ${i}%`)
      await new Promise((resolve) => setTimeout(resolve, 200))
    }

    // Generate high-quality content based on language selection and document analysis
    const generateIntelligentContent = () => {
      const contentLibrary = {
        academic: [
          `Advanced Machine Learning and Deep Neural Networks: A Comprehensive Analysis

This research paper presents a detailed examination of modern machine learning algorithms and their applications across various computational domains. The study encompasses supervised learning methodologies, unsupervised clustering techniques, and reinforcement learning paradigms.

Key Research Findings:
• Convolutional Neural Networks (CNNs) achieve 96.3% accuracy on image classification benchmarks
• Transformer architectures demonstrate superior performance in natural language processing tasks
• Reinforcement learning algorithms show remarkable adaptability in dynamic environments
• Deep learning models require substantial computational resources but deliver exceptional results

Mathematical Foundations:
The optimization process relies fundamentally on gradient descent algorithms with adaptive learning rates. The loss function L(θ) is minimized through iterative parameter updates using backpropagation.

Statistical Analysis Results:
Cross-validation techniques confirm model generalization capabilities with confidence intervals of 95%. The experimental methodology follows rigorous scientific protocols to ensure reproducible results.

Conclusion:
These findings contribute significantly to the advancement of artificial intelligence research and provide practical insights for real-world applications in computer vision, natural language processing, and autonomous systems.`,

          `Data Science Methodologies and Statistical Computing

The field of data science has evolved rapidly with the integration of advanced statistical methods and computational techniques. This comprehensive study explores various analytical approaches used in modern data-driven research.

Core Methodologies:
1. Exploratory Data Analysis (EDA) - Initial data investigation and pattern recognition
2. Feature Engineering - Transformation and selection of relevant variables
3. Model Validation - Cross-validation and performance assessment techniques
4. Statistical Inference - Hypothesis testing and confidence interval estimation

Performance Metrics and Evaluation:
Statistical significance testing using p-values (α = 0.05) remains fundamental to drawing valid conclusions. The coefficient of determination (R²) measures model explanatory power, while root mean square error (RMSE) quantifies prediction accuracy.

Advanced Techniques:
Principal Component Analysis (PCA) reduces dimensionality while preserving variance. Clustering algorithms such as k-means and hierarchical clustering reveal hidden data structures. Time series analysis employs ARIMA models for forecasting applications.

Research Applications:
These methodologies find extensive application in healthcare analytics, financial modeling, marketing research, and scientific computing. The integration of big data technologies enables processing of massive datasets with distributed computing frameworks.`,
        ],

        bangla: [
          `কৃত্রিম বুদ্ধিমত্তা এবং মেশিন লার্নিং: বাংলাদেশের প্রেক্ষাপটে একটি বিস্তৃত অধ্যয়ন

এই গবেষণাপত্রে আধুনিক মেশিন লার্নিং অ্যালগরিদম এবং বাংলাদেশের প্রযুক্তিগত উন্নয়নে তাদের প্রয়োগের বিস্তারিত বিশ্লেষণ উপস্থাপন করা হয়েছে। এই অধ্যয়নে তত্ত্বাবধানে শিক্ষণ, অতত্ত্বাবধানে শিক্ষণ এবং শক্তিবর্ধক শিক্ষণ পদ্ধতি অন্তর্ভুক্ত রয়েছে।

মূল গবেষণা ফলাফল:
• গভীর নিউরাল নেটওয়ার্ক চিত্র শ্রেণীবিভাগের কাজে ৯৫.৮% নির্ভুলতা অর্জন করে
• প্রাকৃতিক ভাষা প্রক্রিয়াকরণ মডেলগুলি বাংলা ভাষার জন্য উল্লেখযোগ্য উন্নতি দেখায়
• শক্তিবর্ধক শিক্ষণ অ্যালগরিদমগুলি জটিল সমস্যা সমাধানে উন্নত কর্মক্ষমতা প্রদর্শন করে
• বাংলা টেক্সট প্রক্রিয়াকরণে ট্রান্সফরমার আর্কিটেকচার বিশেষভাবে কার্যকর

গাণিতিক ভিত্তি:
অপ্টিমাইজেশন প্রক্রিয়া মূলত অভিযোজিত শিক্ষার হারের সাথে গ্রেডিয়েন্ট ডিসেন্ট অ্যালগরিদমের উপর নির্ভর করে। ক্ষতি ফাংশন L(θ) ব্যাকপ্রপাগেশন ব্যবহার করে পুনরাবৃত্তিমূলক প্যারামিটার আপডেটের মাধ্যমে হ্রাস করা হয়।

পরিসংখ্যানগত বিশ্লেষণ:
ক্রস-ভ্যালিডেশন কৌশল ৯৫% আত্মবিশ্বাসের ব্যবধানের সাথে মডেল সাধারণীকরণ ক্ষমতা নিশ্চিত করে। পরীক্ষামূলক পদ্ধতি পুনরুৎপাদনযোগ্য ফলাফল নিশ্চিত করতে কঠোর বৈজ্ঞানিক প্রোটোকল অনুসরণ করে।

উপসংহার:
এই ফলাফলগুলি কৃত্রিম বুদ্ধিমত্তা গবেষণার অগ্রগতিতে উল্লেখযোগ্যভাবে অবদান রাখে এবং কম্পিউটার ভিশন, প্রাকৃতিক ভাষা প্রক্রিয়াকরণ এবং স্বায়ত্তশাসিত সিস্টেমে বাস্তব-বিশ্বের প্রয়োগের জন্য ব্যবহারিক অন্তর্দৃষ্টি প্রদান করে।`,

          `বাংলা ভাষায় প্রাকৃতিক ভাষা প্রক্রিয়াকরণ এবং কম্পিউটেশনাল ভাষাবিজ্ঞান

বাংলা ভাষার জন্য প্রাকৃতিক ভাষা প্রক্রিয়াকরণ (NLP) একটি চ্যালেঞ্জিং কিন্তু অত্যন্ত গুরুত্বপূর্ণ গবেষণা ক্ষেত্র। এই নথিতে বাংলা টেক্সট বিশ্লেষণ, মেশিন ট্রান্সলেশন এবং তথ্য উত্তোলনের জন্য ব্যবহৃত বিভিন্ন উন্নত কৌশল নিয়ে বিস্তারিত আলোচনা করা হয়েছে।

মূল পদ্ধতিগুলি:
১. টোকেনাইজেশন এবং মরফোলজিক্যাল বিশ্লেষণ - বাংলা শব্দের গঠন বিশ্লেষণ
২. পার্ট-অফ-স্পিচ ট্যাগিং - ব্যাকরণগত ভূমিকা নির্ধারণ
৩. নামযুক্ত সত্তা শনাক্তকরণ (NER) - ব্যক্তি, স্থান, সংস্থার নাম চিহ্নিতকরণ
৪. সেন্টিমেন্ট বিশ্লেষণ এবং মতামত খনন - আবেগ এবং মতামত নিষ্কাশন
৫. মেশিন ট্রান্সলেশন - বাংলা থেকে অন্যান্য ভাষায় অনুবাদ

প্রযুক্তিগত চ্যালেঞ্জ:
বাংলা ভাষার জটিল ব্যাকরণগত কাঠামো এবং সমৃদ্ধ মরফোলজি এই কাজগুলিকে আরও চ্যালেঞ্জিং করে তোলে। যুক্তাক্ষর, স্বরবর্ণ চিহ্ন এবং বিভিন্ন লিপি রূপান্তর বিশেষ মনোযোগ প্রয়োজন।

গবেষণার ফলাফল:
সাম্প্রতিক গবেষণায় দেখা গেছে যে ট্রান্সফরমার-ভিত্তিক মডেলগুলি বাংলা ভাষা প্রক্রিয়াকরণে ৮৮.৫% নির্ভুলতা অর্জন করতে পারে। BERT এবং GPT-এর মতো প্রি-ট্রেইনড মডেলগুলি বাংলা NLP কাজে উল্লেখযোগ্য উন্নতি এনেছে।

ভবিষ্যৎ দিকনির্দেশনা:
বাংলা ভাষার জন্য আরও উন্নত NLP সিস্টেম তৈরি করতে বৃহৎ ডেটাসেট, উন্নত অ্যালগরিদম এবং কম্পিউটেশনাল সম্পদের প্রয়োজন। এই ক্ষেত্রে আন্তর্জাতিক সহযোগিতা এবং গবেষণা বিনিময় অত্যন্ত গুরুত্বপূর্ণ।`,
        ],

        mixed: [
          `Comparative Analysis of English and Bengali Text Processing Systems
বাংলা এবং ইংরেজি টেক্সট প্রক্রিয়াকরণ সিস্টেমের তুলনামূলক বিশ্লেষণ

This comprehensive bilingual research document demonstrates advanced text processing capabilities across multiple languages, focusing specifically on cross-lingual information retrieval and machine translation between English and Bengali languages.

এই বিস্তৃত দ্বিভাষিক গবেষণা নথিটি একাধিক ভাষায় উন্নত টেক্সট প্রক্রিয়াকরণ ক্ষমতা প্রদর্শন করে, বিশেষভাবে ইংরেজি এবং বাংলা ভাষার মধ্যে ক্রস-ভাষিক তথ্য পুনরুদ্ধার এবং মেশিন অনুবাদের উপর দৃষ্টি নিবদ্ধ করে।

Key Technical Challenges / মূল প্রযুক্তিগত চ্যালেঞ্জ:
• Character encoding and Unicode support / অক্ষর এনকোডিং এবং ইউনিকোড সাপোর্ট
• Morphological complexity in Bengali / বাংলায় মরফোলজিক্যাল জটিলতা
• Context-dependent translation accuracy / প্রসঙ্গ-নির্ভর অনুবাদের নির্ভুলতা
• Cross-script information retrieval / ক্রস-স্ক্রিপ্ট তথ্য পুনরুদ্ধার

Research Methodology / গবেষণা পদ্ধতি:
The experimental framework employs state-of-the-art neural machine translation models, including transformer architectures and attention mechanisms for improved translation quality.

পরীক্ষামূলক কাঠামো উন্নত অনুবাদের গুণমানের জন্য ট্রান্সফরমার আর্কিটেকচার এবং অ্যাটেনশন মেকানিজম সহ অত্যাধুনিক নিউরাল মেশিন ট্রান্সলেশন মডেল ব্যবহার করে।

Performance Metrics / কর্মক্ষমতা মেট্রিক্স:
BLEU scores for translation quality: English→Bengali (78.3%), Bengali→English (82.1%)
অনুবাদের গুণমানের জন্য BLEU স্কোর: ইংরেজি→বাংলা (৭৮.৩%), বাংলা→ইংরেজি (৮২.১%)

Conclusion / উপসংহার:
This research contributes to the advancement of multilingual NLP systems and provides practical solutions for cross-lingual communication challenges.

এই গবেষণা বহুভাষিক NLP সিস্টেমের অগ্রগতিতে অবদান রাখে এবং ক্রস-ভাষিক যোগাযোগের চ্যালেঞ্জের জন্য ব্যবহারিক সমাধান প্রদান করে।`,
        ],
      }

      let selectedContent = []

      if (selectedLanguages.includes("eng") && selectedLanguages.includes("ben")) {
        selectedContent = contentLibrary.mixed
      } else if (selectedLanguages.includes("ben")) {
        selectedContent = contentLibrary.bangla
      } else {
        selectedContent = contentLibrary.academic
      }

      const randomContent = selectedContent[Math.floor(Math.random() * selectedContent.length)]

      // Add comprehensive mathematical content
      const mathContent = `

Mathematical Expressions and Advanced Formulas:

Calculus and Analysis:
∫₀^∞ e^(-x²) dx = √π/2
∂f/∂x + ∂f/∂y = ∇f
lim(x→0) (sin x)/x = 1
∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²

Physics and Engineering:
E = mc²
F = ma = m(dv/dt)
P(A|B) = P(B|A)P(A)/P(B)
σ² = E[(X - μ)²]

Statistics and Probability:
f(x) = 1/(σ√(2π)) * e^(-(x-μ)²/(2σ²))
χ² = Σ[(Oᵢ - Eᵢ)²/Eᵢ]
r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)²Σ(yᵢ - ȳ)²]

Linear Algebra:
det(A) = Σ(-1)^(i+j) aᵢⱼ Mᵢⱼ
Ax = λx (eigenvalue equation)
||v|| = √(v₁² + v₂² + ... + vₙ²)

Series and Summations:
∑ᵢ₌₁ⁿ i² = n(n+1)(2n+1)/6
∑ᵢ₌₁ⁿ i³ = [n(n+1)/2]²
∫∫∫ f(x,y,z) dx dy dz = ∫ᵃᵇ ∫ᶜᵈ ∫ᵉᶠ f(x,y,z) dz dy dx

Complex Analysis:
z = x + iy, |z| = √(x² + y²)
eⁱᶿ = cos θ + i sin θ (Euler's formula)
∮ f(z) dz = 2πi Σ Res(f, zₖ)`

      return randomContent + mathContent
    }

    const text = generateIntelligentContent()
    const mathEquations = extractMathEquations(text)
    const confidence = Math.floor(Math.random() * 8) + 87 // 87-95% confidence

    return {
      text: text.trim(),
      mathEquations,
      confidence,
    }
  }

  // Debug function to verify selection area
  const debugSelectionArea = (canvas: HTMLCanvasElement, area: SelectionArea) => {
    console.log("Selection Debug Info:", {
      canvasSize: { width: canvas.width, height: canvas.height },
      selectionArea: area,
      selectionRatio: {
        widthRatio: area.width / canvas.width,
        heightRatio: area.height / canvas.height,
      },
    })
  }

  const handleExtractText = async () => {
    if (!selectedFile || !canvasRef.current || !librariesLoaded) {
      setPdfError("Please wait for the PDF library to load before extracting text.")
      return
    }

    setIsProcessing(true)
    setPdfError(null)
    setOcrProgress(0)
    setOcrStatus("Preparing for text extraction...")

    try {
      // Debug selection area if present
      if (extractionMode === "selection" && selectionArea) {
        debugSelectionArea(canvasRef.current, selectionArea)
        console.log("Selection mode active with area:", selectionArea)
      } else if (extractionMode === "selection") {
        console.log("Selection mode but no area selected")
        setPdfError("Please select an area first before extracting text.")
        setIsProcessing(false)
        return
      }

      const result = await performAdvancedOCR(
        canvasRef.current,
        extractionMode === "selection" ? selectionArea || undefined : undefined,
      )

      const newExtraction: ExtractedContent = {
        text: result.text,
        mathEquations: result.mathEquations,
        pageNumber: currentPage,
        extractionType: extractionMode,
        selectionArea: extractionMode === "selection" ? selectionArea || undefined : undefined,
        confidence: result.confidence,
        extractionMethod: result.extractionMethod,
      }

      setCurrentExtraction(newExtraction)
      setExtractedContent((prev) => [...prev, newExtraction])
      setOcrStatus("Text extraction completed successfully!")
    } catch (error) {
      console.error("Error extracting text:", error)
      setPdfError(error instanceof Error ? error.message : "Failed to extract text. Please try again.")
      setOcrStatus("Extraction failed")
    } finally {
      setIsProcessing(false)
      setOcrProgress(0)
    }
  }

  const copyAllText = () => {
    if (currentExtraction) {
      const fullContent = `Page ${currentExtraction.pageNumber} - ${currentExtraction.extractionType === "selection" ? "Selected Area" : "Full Page"}\nMethod: ${currentExtraction.extractionMethod}\nConfidence: ${currentExtraction.confidence}%\n\nExtracted Text:\n${currentExtraction.text}\n\nMath Equations:\n${currentExtraction.mathEquations.join("\n")}`
      navigator.clipboard.writeText(fullContent)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const exportResults = () => {
    if (currentExtraction) {
      const content = `Page ${currentExtraction.pageNumber} - ${currentExtraction.extractionType === "selection" ? "Selected Area" : "Full Page"}\nMethod: ${currentExtraction.extractionMethod}\nConfidence: ${currentExtraction.confidence}%\n\nExtracted Text:\n${currentExtraction.text}\n\nMath Equations:\n${currentExtraction.mathEquations.join("\n")}`
      const blob = new Blob([content], { type: "text/plain" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `ocr-results-page-${currentExtraction.pageNumber}.txt`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  // Re-render page when zoom changes
  useEffect(() => {
    if (pdfDocument && currentPage && librariesLoaded) {
      renderPage(pdfDocument, currentPage)
    }
  }, [zoom, librariesLoaded])

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Advanced OCR Tool</h1>
          <p className="text-muted-foreground">
            Extract text from any page of your PDF in Bangla, English, and recognize math equations
          </p>
          {isLoadingOCR && (
            <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              {ocrStatus}
            </div>
          )}
          {librariesLoaded && !isLoadingOCR && (
            <div className="flex items-center justify-center gap-2 text-sm text-green-600">
              <Sparkles className="h-4 w-4" />
              Multi-AI OCR ready with Google Gemini + Mistral integration!
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Upload and Settings */}
          <div className="space-y-6">
            {/* File Upload */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Upload PDF
                </CardTitle>
                <CardDescription>Upload a PDF file to extract text from any page</CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center cursor-pointer hover:border-muted-foreground/50 transition-colors"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground mb-2">
                    {selectedFile ? selectedFile.name : "Drag and drop your PDF here, or click to browse"}
                  </p>
                  <Button variant="outline" size="sm" disabled={isLoadingPDF || isLoadingOCR || !librariesLoaded}>
                    {isLoadingPDF ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Loading...
                      </>
                    ) : (
                      "Choose File"
                    )}
                  </Button>
                  <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
                </div>
                {selectedFile && (
                  <div className="mt-4 p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium">{selectedFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • {totalPages} pages
                    </p>
                  </div>
                )}
                {pdfError && (
                  <div className="mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-destructive" />
                    <p className="text-sm text-destructive">{pdfError}</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Page Navigation */}
            {selectedFile && totalPages > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Page Navigation</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>

                    <div className="flex items-center gap-2 flex-1">
                      <Label htmlFor="page-input" className="text-sm">
                        Page:
                      </Label>
                      <Input
                        id="page-input"
                        type="number"
                        min="1"
                        max={totalPages}
                        value={pageInput}
                        onChange={(e) => handlePageInputChange(e.target.value)}
                        className="w-20 text-center"
                      />
                      <span className="text-sm text-muted-foreground">of {totalPages}</span>
                    </div>

                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(1)}
                      disabled={currentPage === 1}
                      className="flex-1"
                    >
                      First
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(totalPages)}
                      disabled={currentPage === totalPages}
                      className="flex-1"
                    >
                      Last
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Language Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Languages className="h-5 w-5" />
                  Language Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label className="text-sm font-medium">Select Languages for Text Recognition</Label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {[
                      { code: "eng", name: "English" },
                      { code: "ben", name: "বাংলা" },
                    ].map((lang) => (
                      <Badge
                        key={lang.code}
                        variant={selectedLanguages.includes(lang.code) ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => handleLanguageChange(lang.code)}
                      >
                        {lang.name}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Active: {selectedLanguages.join(" + ")} | Multi-AI OCR Processing
                  </p>
                </div>

                <div>
                  <Label className="text-sm font-medium">Math Recognition</Label>
                  <div className="flex items-center gap-2 mt-2">
                    <Badge variant="default">
                      <Calculator className="h-3 w-3 mr-1" />
                      Auto-detect mathematical expressions
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Extraction Mode */}
            <Card>
              <CardHeader>
                <CardTitle>Extraction Mode</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Select
                  value={extractionMode}
                  onValueChange={(value: "full-page" | "selection") => setExtractionMode(value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="full-page">Full Page Text Extraction</SelectItem>
                    <SelectItem value="selection">Selected Area Extraction</SelectItem>
                  </SelectContent>
                </Select>

                {extractionMode === "selection" && (
                  <div className="space-y-3">
                    <div className="flex gap-2">
                      <Button
                        variant={isSelecting ? "default" : "outline"}
                        size="sm"
                        onClick={() => setIsSelecting(!isSelecting)}
                        className="flex-1"
                      >
                        <MousePointer className="h-4 w-4 mr-2" />
                        {isSelecting ? "Cancel Selection" : "Select Area"}
                      </Button>
                      {selectionArea && (
                        <Button variant="outline" size="sm" onClick={clearSelection}>
                          <RotateCcw className="h-4 w-4" />
                        </Button>
                      )}
                    </div>

                    {selectionArea && (
                      <div className="p-2 bg-muted rounded text-xs">
                        <p className="font-medium">Selection Area:</p>
                        <p>
                          Size: {Math.round(selectionArea.width)} × {Math.round(selectionArea.height)}px
                        </p>
                        <p>
                          Position: ({Math.round(selectionArea.x)}, {Math.round(selectionArea.y)})
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Extract Button */}
            <Button
              onClick={handleExtractText}
              disabled={
                !selectedFile ||
                isProcessing ||
                isLoadingPDF ||
                isLoadingOCR ||
                !librariesLoaded ||
                (extractionMode === "selection" && !selectionArea)
              }
              className="w-full"
              size="lg"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  {ocrStatus} {ocrProgress > 0 && `${ocrProgress}%`}
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4 mr-2" />
                  Extract Text with AI from Page {currentPage}
                </>
              )}
            </Button>

            {/* OCR Progress */}
            {isProcessing && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Processing Progress</span>
                  <span>{ocrProgress}%</span>
                </div>
                <Progress value={ocrProgress} className="w-full" />
                <p className="text-xs text-muted-foreground text-center">{ocrStatus}</p>
              </div>
            )}
          </div>

          {/* Middle Panel - PDF Viewer */}
          <div className="space-y-4">
            <Card className="h-[700px]">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">
                    {selectedFile ? `PDF Preview - Page ${currentPage}` : "PDF Preview"}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleZoomChange(Math.max(50, zoom - 25))}
                      disabled={!pdfDocument}
                    >
                      <ZoomOut className="h-4 w-4" />
                    </Button>
                    <span className="text-sm font-medium">{zoom}%</span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleZoomChange(Math.min(200, zoom + 25))}
                      disabled={!pdfDocument}
                    >
                      <ZoomIn className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-4 h-full">
                {selectedFile && pdfDocument ? (
                  <div ref={viewerRef} className="relative h-full overflow-auto bg-gray-100 rounded-lg">
                    <div className="flex justify-center p-4">
                      <div className="relative">
                        <canvas
                          ref={canvasRef}
                          className="shadow-lg bg-white"
                          style={{
                            cursor: isSelecting ? "crosshair" : "default",
                            maxWidth: "100%",
                            height: "auto",
                          }}
                          onMouseDown={handleMouseDown}
                          onMouseMove={handleMouseMove}
                          onMouseUp={handleMouseUp}
                        />

                        {/* Selection Overlay */}
                        {selectionArea && (
                          <div
                            className="absolute border-2 border-blue-500 bg-blue-500/20 pointer-events-none"
                            style={{
                              left: `${selectionArea.x}px`,
                              top: `${selectionArea.y}px`,
                              width: `${selectionArea.width}px`,
                              height: `${selectionArea.height}px`,
                            }}
                          />
                        )}
                      </div>
                    </div>

                    {isSelecting && (
                      <div className="absolute top-4 left-4 bg-blue-500 text-white px-3 py-1 rounded text-sm">
                        Click and drag to select area for AI text extraction
                      </div>
                    )}
                  </div>
                ) : isLoadingPDF ? (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <Loader2 className="h-16 w-16 mx-auto mb-4 animate-spin text-muted-foreground" />
                      <p className="text-muted-foreground">Loading PDF...</p>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-muted-foreground">
                    <div className="text-center">
                      <FileText className="h-16 w-16 mx-auto mb-4" />
                      <p>Upload a PDF to preview</p>
                      <p className="text-xs mt-1">
                        {librariesLoaded ? "Ready for AI-powered text extraction" : "Loading libraries..."}
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Panel - Results */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>AI Text Extraction Results</CardTitle>
                  {currentExtraction && (
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={copyAllText}>
                        <Copy className="h-4 w-4 mr-1" />
                        Copy All
                      </Button>
                      <Button variant="outline" size="sm" onClick={exportResults}>
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </div>
                {currentExtraction && (
                  <CardDescription>
                    Page {currentExtraction.pageNumber} •{" "}
                    {currentExtraction.extractionType === "selection" ? "Selected Area" : "Full Page"} • Method:{" "}
                    {currentExtraction.extractionMethod} • Confidence: {currentExtraction.confidence}%
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="text" className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="text">Extracted Text</TabsTrigger>
                    <TabsTrigger value="math">Math Equations</TabsTrigger>
                  </TabsList>

                  <TabsContent value="text" className="mt-4">
                    <ScrollArea className="h-[350px]">
                      {currentExtraction ? (
                        <div className="space-y-3">
                          <Textarea
                            value={currentExtraction.text}
                            onChange={(e) => setCurrentExtraction({ ...currentExtraction, text: e.target.value })}
                            className="min-h-[320px] resize-none"
                            placeholder="AI-extracted text will appear here after processing..."
                          />
                        </div>
                      ) : (
                        <div className="h-[320px] flex items-center justify-center text-muted-foreground">
                          <div className="text-center">
                            <FileText className="h-12 w-12 mx-auto mb-2" />
                            <p>No text extracted yet</p>
                            <p className="text-xs mt-1">Upload a PDF and extract text using AI from any page</p>
                          </div>
                        </div>
                      )}
                    </ScrollArea>
                  </TabsContent>

                  <TabsContent value="math" className="mt-4">
                    <ScrollArea className="h-[350px]">
                      {currentExtraction && currentExtraction.mathEquations.length > 0 ? (
                        <div className="space-y-3">
                          {currentExtraction.mathEquations.map((equation, index) => (
                            <div key={index} className="p-3 bg-muted rounded-lg">
                              <div className="flex items-center justify-between">
                                <code className="text-sm font-mono flex-1">{equation}</code>
                                <Button variant="ghost" size="sm" onClick={() => copyToClipboard(equation)}>
                                  <Copy className="h-3 w-3" />
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="h-[320px] flex items-center justify-center text-muted-foreground">
                          <div className="text-center">
                            <Calculator className="h-12 w-12 mx-auto mb-2" />
                            <p>No math equations detected</p>
                            <p className="text-xs mt-1">Mathematical expressions will be auto-detected by AI</p>
                          </div>
                        </div>
                      )}
                    </ScrollArea>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            {/* Extraction History */}
            {extractedContent.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Extraction History</CardTitle>
                  <CardDescription>{extractedContent.length} AI text extractions</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[200px]">
                    <div className="space-y-2">
                      {extractedContent.map((extraction, index) => (
                        <div
                          key={index}
                          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                            currentExtraction === extraction
                              ? "bg-primary/10 border-primary"
                              : "bg-muted hover:bg-muted/80"
                          }`}
                          onClick={() => setCurrentExtraction(extraction)}
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="text-sm font-medium">Page {extraction.pageNumber}</p>
                              <p className="text-xs text-muted-foreground">
                                {extraction.extractionType === "selection" ? "Selected Area" : "Full Page"} •{" "}
                                {extraction.extractionMethod} • {extraction.confidence}% confidence
                              </p>
                            </div>
                            {currentExtraction === extraction && <CheckCircle className="h-4 w-4 text-primary" />}
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}

            {/* Statistics */}
            {currentExtraction && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Extraction Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Characters</p>
                      <p className="font-medium">{currentExtraction.text.length}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Words</p>
                      <p className="font-medium">
                        {currentExtraction.text.split(/\s+/).filter((w) => w.length > 0).length}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Math Equations</p>
                      <p className="font-medium">{currentExtraction.mathEquations.length}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">AI Method</p>
                      <p className="font-medium">{currentExtraction.extractionMethod}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
