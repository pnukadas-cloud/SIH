"""
MAITRI — Demonstration Flow & Presentation Guide PDF Generator
Generates a publication-grade, multi-page vector PDF detailing the complete
step-by-step demonstration walkthrough, visual cues, voiceover script, and judge Q&A.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 752, "ISRO / BHARTIYA ANTARIKSH STATION · MAITRI PROJECT DEMONSTRATION GUIDE")
            self.drawRightString(letter[0] - 54, 752, "SIH 2025 · Problem ID 25175")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 745, letter[0] - 54, 745)
            
        # Running Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY · ISRO / DEPARTMENT OF SPACE")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0] - 54, 48)
        self.restoreState()


def build_pdf(filename="MAITRI_Demonstration_and_Presentation_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0F172A")    # Dark Slate Navy
    c_accent = colors.HexColor("#4F46E5")     # Indigo-600
    c_cyan = colors.HexColor("#0284C7")       # Sky-600
    c_emerald = colors.HexColor("#059669")    # Emerald-600
    c_amber = colors.HexColor("#D97706")      # Amber-600
    c_text = colors.HexColor("#1E293B")       # Slate-800
    c_subtext = colors.HexColor("#475569")    # Slate-600
    c_border = colors.HexColor("#E2E8F0")     # Slate-200
    c_bg_light = colors.HexColor("#F8FAFC")   # Slate-50

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=c_primary,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=c_accent,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=c_accent,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=c_text,
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        "Callout_Text",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_text
    )

    script_spoken_style = ParagraphStyle(
        "Script_Spoken",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#090D16")
    )

    label_bold = ParagraphStyle(
        "LabelBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=c_primary
    )

    story = []

    # -------------------------------------------------------------------------
    # COVER / HEADER BANNER
    # -------------------------------------------------------------------------
    meta_table_data = [
        [
            Paragraph("<b>ISRO / DEPARTMENT OF SPACE · GAGANYAAN & BAS EXPEDITION</b>", ParagraphStyle("HdrMeta", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=c_cyan)),
            Paragraph("<b>SMART INDIA HACKATHON 2025</b>", ParagraphStyle("HdrMetaR", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=c_accent, alignment=2))
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[330, 174])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("MAITRI — Project Demonstration & Presentation Guide", title_style))
    story.append(Paragraph("End-to-End Walkthrough Script, Visual Flow, System Narrative & Evaluation Defense", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=2, spaceAfter=14))

    # Executive Overview Box
    overview_text = (
        "<b>Mission Brief:</b> This document provides the complete, chronological demonstration script "
        "and presentation walkthrough for the <b>MAITRI</b> (Multimodal AI Assistant for Astronaut Psychological & "
        "Physical Well-Being) system. Designed specifically for video recording, live judge defense, or evaluator demonstrations, "
        "it covers on-screen visual cues, exact voiceover narrative, technical highlights, and high-frequency evaluators' questions."
    )
    ov_table = Table([[Paragraph(overview_text, callout_style)]], colWidths=[504])
    ov_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_accent),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 12))

    # Key Specifications Table
    spec_data = [
        [Paragraph("<b>Parameter</b>", label_bold), Paragraph("<b>Specification / Recommendation</b>", label_bold)],
        [Paragraph("Target Video Duration", body_style), Paragraph("<b>3 to 5 Minutes</b> (Ideal for hackathons and technical reviews)", body_style)],
        [Paragraph("Recommended Resolution", body_style), Paragraph("<b>1080p Full HD (1920 × 1080)</b>, 60 FPS in Full-Screen (Press F11)", body_style)],
        [Paragraph("Audio Setup", body_style), Paragraph("Crisp microphone voiceover with calm, confident cadence", body_style)],
        [Paragraph("Primary URL", body_style), Paragraph("<b>http://127.0.0.1:8000/</b> (Landing Page -> Login -> Dashboard)", body_style)],
        [Paragraph("Demo Credentials", body_style), Paragraph("<b>AST-001 (Aryan)</b> · Passcode: <b>password123</b> (or 1-click test chips)", body_style)],
    ]
    spec_table = Table(spec_data, colWidths=[160, 344])
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2FF")),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(spec_table)
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------------------
    # DEMONSTRATION TIMELINE OVERVIEW
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Demonstration Structure & Timeline", h1_style))
    
    timeline_data = [
        [Paragraph("<b>Timestamp</b>", label_bold), Paragraph("<b>Stage / Route</b>", label_bold), Paragraph("<b>Core Objective</b>", label_bold)],
        [Paragraph("<b>0:00 – 0:45</b>", body_style), Paragraph("Landing Page (<code>/</code>)", body_style), Paragraph("Mission context, Gaganyaan/BAS problem statement, 3D Torus, 4 core pillars", body_style)],
        [Paragraph("<b>0:45 – 1:30</b>", body_style), Paragraph("Login Portal (<code>/login</code>)", body_style), Paragraph("Crew authentication, high-contrast inputs, Face ID optical biometric scan", body_style)],
        [Paragraph("<b>1:30 – 2:45</b>", body_style), Paragraph("Astronaut HUD (<code>/dashboard</code>)", body_style), Paragraph("Optical standby logic, PERCLOS fatigue detection, 5 flight simulation scenarios", body_style)],
        [Paragraph("<b>2:45 – 3:45</b>", body_style), Paragraph("Subsystem Deep-Dive", body_style), Paragraph("Analysis FACS units, Interventions, Ground Alerts, Flight Surgeon Console", body_style)],
        [Paragraph("<b>3:45 – 4:30</b>", body_style), Paragraph("AI Companion Chat", body_style), Paragraph("Live contextual dialogue on spaceflight anxiety, grounding protocol response", body_style)],
        [Paragraph("<b>4:30 – 5:00</b>", body_style), Paragraph("Session Close & Conclusion", body_style), Paragraph("Crew session logout, summary of offline edge resilience, and closing statement", body_style)],
    ]
    timeline_table = Table(timeline_data, colWidths=[84, 150, 270])
    timeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2FF")),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(timeline_table)
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------------------
    # SCENE-BY-SCENE SCRIPT
    # -------------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("2. Scene-by-Scene Demonstration Script", h1_style))
    story.append(Paragraph("Follow this chronological narrative while performing the visual interactions on screen:", body_style))
    story.append(Spacer(1, 8))

    def make_scene_block(scene_title, timing, route, visual_steps, script_text):
        content = []
        header_text = f"<b>{scene_title}</b> ({timing}) — <i>Route: {route}</i>"
        content.append(Paragraph(header_text, h2_style))
        
        # Visual cues
        cues_para = Paragraph(f"<b>Visual Actions:</b> {visual_steps}", body_style)
        content.append(cues_para)
        content.append(Spacer(1, 4))
        
        # Script box
        script_box_data = [[
            Paragraph(f"<b>Spoken Narrative (Voiceover):</b><br/>\"{script_text}\"", script_spoken_style)
        ]]
        s_box = Table(script_box_data, colWidths=[504])
        s_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
            ('BOX', (0,0), (-1,-1), 1, c_cyan),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        content.append(s_box)
        content.append(Spacer(1, 12))
        return KeepTogether(content)

    # Scene 1
    story.append(make_scene_block(
        "Scene 1: The Problem & The Landing Page",
        "0:00 – 0:45",
        "/",
        "Start on full-screen landing page. Pause 2 seconds on the rotating 3D torus. Scroll gently over the 4 feature cards (Multimodal, AI Intervention, Crew Well-Being, Ground Alerting). Click 'Enter MAITRI'.",
        "Hello everyone. In long-duration spaceflight—such as India's Gaganyaan mission and the upcoming Bhartiya Antariksh Station—astronauts face intense confinement, microgravity, and cognitive stress. Traditional clinical monitoring is impossible during orbital blackout periods. This is MAITRI: a multimodal AI companion that continuously monitors crew health through non-invasive optical, acoustic, and linguistic signals. Built for edge deployment, it operates 100% offline without external dependencies."
    ))

    # Scene 2
    story.append(make_scene_block(
        "Scene 2: Mission Crew Authentication & Face ID",
        "0:45 – 1:30",
        "/login",
        "Land on centered glassmorphism card. Click 'AST-001 (Aryan)' test fill chip to show crisp white inputs with bold black text. Click the eye icon to reveal the passcode. Click 'Face ID Biometric Login' to preview the optical reticle modal. Close modal and click 'Login to Station'.",
        "Security and data privacy are paramount in mission control. MAITRI features role-based access control with SQLite database persistence. Astronauts can log in with credentials or use our zero-cloud Face ID Biometric Login, which extracts 128-dimensional Local Binary Pattern facial vectors to instantly verify identity on edge hardware. Let's log in as Mission Commander Aryan."
    ))

    # Scene 3
    story.append(make_scene_block(
        "Scene 3: Astronaut HUD & Live Scenarios",
        "1:30 – 2:45",
        "/dashboard",
        "Point to Aryan's dynamic profile and live MET clock. Show that when camera is idle, HUD remains in clean 'Optical Standby' (-- / 100). Next, click '2. Docking Stress' (watch gauge jump to Stressed). Click '4. Severe Fatigue' (point out PERCLOS > 12% alert). Click '1. Nominal Orbit' to return to Level 0.",
        "Upon authentication, MAITRI initializes an isolated monitoring session loaded with Aryan's personalized baseline vitals. Notice our optical reticle: when sensors are idle, the system stays in an intelligent standby state, preventing false fatigue alarms. When optical tracking engages, our engine evaluates 7 affective states alongside PERCLOS, blink rate, and yawn frequency. Using our simulation controls: triggering Docking Stress elevates cognitive strain, while Severe Fatigue immediately flags prolonged eye closure."
    ))

    # Scene 4
    story.append(make_scene_block(
        "Scene 4: Analysis, Interventions & Flight Surgeon Console",
        "2:45 – 3:45",
        "Tabs: Analysis, Interventions, Alerts, Surgeon",
        "Click 'Analysis' (show Action Units & pitch). Click 'Interventions' (show 4-7-8 breathing and relaxation cards). Click 'Alerts' (show queued ground passes). Click 'Switch to Surgeon' (show multi-crew medical readiness triage).",
        "MAITRI doesn't just diagnose; it provides autonomous closed-loop support. Under Analysis, we track facial action units and acoustic prosody over time. Under Interventions, MAITRI prescribes adaptive grounding exercises, like guided box breathing, before stress escalates. Critical anomalies are queued for synchronization with ISRO ground stations during scheduled orbital passes, and the Flight Surgeon Console gives ground medical teams instantaneous crew oversight."
    ))

    # Scene 5 & 6
    story.append(make_scene_block(
        "Scene 5: Interactive AI Companion Dialogue",
        "3:45 – 4:30",
        "/dashboard (Chat)",
        "Scroll to the AI Companion Chat. Type: 'I feel a bit overwhelmed by the docking thruster alarms.' Press Enter and highlight MAITRI's prompt procedural and psychological grounding response.",
        "Astronauts also have a 24/7 conversational companion. When an astronaut reports feeling overwhelmed by docking alarms, MAITRI provides psychological reassurance alongside procedural clarity—explaining how relative-navigation LiDAR and motorized capture hooks ensure safety. Everything runs locally with zero external latency, with an optional 1-click connector to Gemini 1.5 Flash when satellite uplink is active."
    ))

    story.append(make_scene_block(
        "Scene 6: Session Lifecycle & Conclusion",
        "4:30 – 5:00",
        "/dashboard -> /login -> /",
        "Click 'Logout' in top header (smooth redirect to /login). Click 'Back to Home' (return to 3D Torus Landing Page). Deliver closing statement with confidence.",
        "To summarize: MAITRI provides a non-invasive, privacy-preserving, and mission-resilient well-being infrastructure for human spaceflight. By fusing real-time multimodal sensing, automated interventions, and ground telemetry synchronization, MAITRI ensures our astronauts remain physically resilient and mentally grounded. Thank you, and Jai Hind!"
    ))

    # -------------------------------------------------------------------------
    # JUDGE FAQ & TECHNICAL DEFENSE
    # -------------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Evaluator Defense & Judge FAQ", h1_style))
    story.append(Paragraph("Key questions judges frequently ask and how to answer them concisely:", body_style))
    story.append(Spacer(1, 6))

    qa_list = [
        ("Q1: How does MAITRI handle false fatigue positives when the camera is off or lighting changes?",
         "MAITRI employs an explicit optical standby state. When no face or acoustic stream is active, the well-being evaluator outputs a clean standby state (-- / 100, 0 pts) rather than assuming nominal or fatigue. Furthermore, PERCLOS evaluation requires a minimum 15-frame rolling window and authentic yawns strictly require the absence of smiling (AU12) and Mouth Aspect Ratio > 0.68."),
        
        ("Q2: Can this run completely offline without internet on the spacecraft?",
         "Yes. All core subsystems—OpenCV multi-scale Haar cascades, Facial Emotion Recognition, speech acoustics, SQLite session database, and the cognitive dialogue engine—run 100% locally on edge hardware. External LLMs (Google Gemini 1.5 Flash) are strictly optional and only activated when satellite communication bandwidth is available."),
         
        ("Q3: How is astronaut privacy maintained between crew members?",
         "MAITRI implements strict session isolation in SQLite. Monitoring sessions are keyed to unique session tokens and astronaut IDs. Astronaut A's affective telemetry, journal entries, and dialogue records can never be queried or viewed by Astronaut B, adhering to ISRO flight medicine confidentiality standards."),
         
        ("Q4: What happens if optical face tracking fails due to extreme head pose?",
         "The computer vision pipeline includes multi-spectral skin chrominance verification and temporal exponential moving average (EMA) smoothing. If optical lock is temporarily lost, the system holds the last verified baseline state for 3 frames before declaring a search state, completely eliminating frame fluttering.")
    ]

    for q, a in qa_list:
        qa_data = [
            [Paragraph(f"<b>{q}</b>", ParagraphStyle("QA_Q", fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=c_primary))],
            [Paragraph(a, ParagraphStyle("QA_A", fontName="Helvetica", fontSize=9, leading=13, textColor=c_text))]
        ]
        qa_tbl = Table(qa_data, colWidths=[504])
        qa_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2FF")),
            ('BACKGROUND', (0,1), (-1,1), c_bg_light),
            ('BOX', (0,0), (-1,-1), 0.5, c_border),
            ('PADDING', (0,0), (-1,-1), 7),
        ]))
        story.append(qa_tbl)
        story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # PRO-TIPS FOR RECORDING
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 6))
    story.append(Paragraph("4. Recording Best Practices Checklist", h1_style))
    tips_data = [
        [Paragraph("<b>Step</b>", label_bold), Paragraph("<b>Action Item</b>", label_bold)],
        [Paragraph("1. Browser Mode", body_style), Paragraph("Press <b>F11</b> in Chrome/Edge to enter full-screen mode to hide URL bar and bookmarks.", body_style)],
        [Paragraph("2. Cursor Pace", body_style), Paragraph("Move cursor smoothly and deliberately between UI cards; avoid quick, erratic gestures.", body_style)],
        [Paragraph("3. Screen Recorder", body_style), Paragraph("Use <b>OBS Studio</b> (recommended at 1080p, 60fps) or <b>Windows Game Bar (Win + G)</b>.", body_style)],
        [Paragraph("4. Server Readiness", body_style), Paragraph("Verify <code>uvicorn server:app --port 8000</code> is active before launching the recorder.", body_style)],
        [Paragraph("5. Scenario Transitions", body_style), Paragraph("Pause for 2 seconds after triggering scenarios (e.g. Docking Stress) to let gauges animate.", body_style)]
    ]
    tips_table = Table(tips_data, colWidths=[120, 384])
    tips_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2FF")),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(tips_table)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Generated {filename} successfully.")

if __name__ == "__main__":
    out_pdf = "MAITRI_Demonstration_and_Presentation_Guide.pdf"
    if len(sys.argv) > 1:
        out_pdf = sys.argv[1]
    build_pdf(out_pdf)
