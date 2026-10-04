// Build the IEEE-format Word paper on menstrual health and hygiene.
//
// Layout mirrors the supplied IEEE conference template: US Letter, Times New
// Roman, single-column title/author block, then two 3.5-in columns with a
// 0.25-in gap; small-caps Roman-numeral section headings, italic lettered
// subheadings, 8-pt captions and references.
//
// Run:  python make_figures.py && node build_paper.js

const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, FootnoteReferenceRun, ImageRun,
  LevelFormat, Packer, Paragraph, SectionType, Tab, TabStopType, Table,
  TableCell, TableLayoutType, TableRow, TextRun, VerticalAlign, WidthType,
} = require("docx");

const HERE = __dirname;
const OUT = path.join(HERE, "Menstrual_Health_and_Hygiene_IEEE_Paper.docx");

const FONT = "Times New Roman";
const PAGE = { width: 12240, height: 15840 };
const MARGIN = { top: 1080, bottom: 1440, left: 900, right: 900, header: 720, footer: 720 };
const COL_GAP = 360;
const COL_W = (PAGE.width - MARGIN.left - MARGIN.right - COL_GAP) / 2; // 5040
const FULL_W = PAGE.width - MARGIN.left - MARGIN.right; // 10440

// ---------------------------------------------------------------------------
// References: numbered automatically in order of first citation.
// ---------------------------------------------------------------------------
const REFS = {
  worldbank: [["World Bank, “Menstrual health and hygiene,” World Bank Brief, May 2022. [Online]. Available: https://www.worldbank.org/en/topic/water/brief/menstrual-health-and-hygiene"]],
  unicef2019: [["UNICEF, "], ["Guidance on Menstrual Health and Hygiene", "i"], [". New York, NY, USA: UNICEF, 2019."]],
  hennegan2021: [["J. Hennegan "], ["et al.", "i"], [", “Menstrual health: A definition for policy, practice, and research,” "], ["Sexual and Reproductive Health Matters", "i"], [", vol. 29, no. 1, Art. no. 1911618, 2021."]],
  sommer2015: [["M. Sommer, J. S. Hirsch, C. Nathanson, and R. G. Parker, “Comfortably, safely, and without shame: Defining menstrual hygiene management as a public health issue,” "], ["American Journal of Public Health", "i"], [", vol. 105, no. 7, pp. 1302–1311, 2015."]],
  plan2017: [["Plan International UK, “1 in 10 girls have been unable to afford sanitary wear,” survey press release, London, U.K., Oct. 2017."]],
  hennegan2019: [["J. Hennegan "], ["et al.", "i"], [", “Women’s and girls’ experiences of menstruation in low- and middle-income countries: A systematic review and qualitative metasynthesis,” "], ["PLoS Medicine", "i"], [", vol. 16, no. 5, Art. no. e1002803, 2019."]],
  un2015: [["United Nations General Assembly, “Transforming our world: The 2030 Agenda for Sustainable Development,” Resolution A/RES/70/1, Oct. 2015."]],
  jmp2012: [["WHO/UNICEF Joint Monitoring Programme, “Consultation on draft long list of goal, target and indicator options for future global monitoring of water, sanitation and hygiene,” WHO and UNICEF, 2012."]],
  munro2018: [["M. G. Munro, H. O. D. Critchley, and I. S. Fraser, “The two FIGO systems for normal and abnormal uterine bleeding symptoms and classification of causes of abnormal uterine bleeding in the reproductive years: 2018 revisions,” "], ["International Journal of Gynecology & Obstetrics", "i"], [", vol. 143, no. 3, pp. 393–408, 2018."]],
  iacovides2015: [["S. Iacovides, I. Avidon, and F. C. Baker, “What we know about primary dysmenorrhea today: A critical review,” "], ["Human Reproduction Update", "i"], [", vol. 21, no. 6, pp. 762–778, 2015."]],
  whoendo: [["World Health Organization, “Endometriosis,” WHO Fact Sheet, Geneva, Switzerland, Mar. 2023."]],
  whopcos: [["World Health Organization, “Polycystic ovary syndrome,” WHO Fact Sheet, Geneva, Switzerland, Jun. 2023."]],
  jmp2023: [["WHO and UNICEF, "], ["Progress on Household Drinking Water, Sanitation and Hygiene 2000–2022: Special Focus on Gender", "i"], [". New York, NY, USA: UNICEF and WHO, 2023."]],
  jmp2024: [["WHO and UNICEF, "], ["Progress on Drinking Water, Sanitation and Hygiene in Schools 2015–2023: Special Focus on Menstrual Health", "i"], [". New York, NY, USA: UNICEF and WHO, 2024."]],
  nfhs4: [["International Institute for Population Sciences (IIPS) and ICF, "], ["National Family Health Survey (NFHS-4), 2015–16: India", "i"], [". Mumbai, India: IIPS, 2017."]],
  nfhs5: [["International Institute for Population Sciences (IIPS) and ICF, "], ["National Family Health Survey (NFHS-5), 2019–21: India", "i"], [", vol. I. Mumbai, India: IIPS, 2021."]],
  vaneijk2016: [["A. M. van Eijk "], ["et al.", "i"], [", “Menstrual hygiene management among adolescent girls in India: A systematic review and meta-analysis,” "], ["BMJ Open", "i"], [", vol. 6, no. 3, Art. no. e010290, 2016."]],
  das2015: [["P. Das "], ["et al.", "i"], [", “Menstrual hygiene practices, WASH access and the risk of urogenital infection in women from Odisha, India,” "], ["PLoS ONE", "i"], [", vol. 10, no. 6, Art. no. e0130777, 2015."]],
  sumpter2013: [["C. Sumpter and B. Torondel, “A systematic review of the health and social effects of menstrual hygiene management,” "], ["PLoS ONE", "i"], [", vol. 8, no. 4, Art. no. e62004, 2013."]],
  toxics2021: [["Toxics Link, “Menstrual products and their disposal,” New Delhi, India, 2021."]],
  vaneijk2019: [["A. M. van Eijk "], ["et al.", "i"], [", “Menstrual cup use, leakage, acceptability, safety, and availability: A systematic review and meta-analysis,” "], ["The Lancet Public Health", "i"], [", vol. 4, no. 8, pp. e376–e393, 2019."]],
  phillips2016: [["P. A. Phillips-Howard "], ["et al.", "i"], [", “Menstrual cups and sanitary pads to reduce school attrition, and sexually transmitted and reproductive tract infections: A cluster randomised controlled feasibility study in rural Western Kenya,” "], ["BMJ Open", "i"], [", vol. 6, no. 11, Art. no. e013229, 2016."]],
  montgomery2016: [["P. Montgomery, J. Hennegan, C. Dolan, M. Wu, L. Steinfield, and L. Scott, “Menstruation and the cycle of poverty: A cluster quasi-randomised control trial of sanitary pad and puberty education provision in Uganda,” "], ["PLoS ONE", "i"], [", vol. 11, no. 12, Art. no. e0166122, 2016."]],
  hennegan2016: [["J. Hennegan and P. Montgomery, “Do menstrual hygiene management interventions improve education and psychosocial outcomes for women and girls in low and middle income countries? A systematic review,” "], ["PLoS ONE", "i"], [", vol. 11, no. 2, Art. no. e0146985, 2016."]],
  mohfw2011: [["Ministry of Health and Family Welfare, Government of India, “Scheme for promotion of menstrual hygiene among adolescent girls in rural areas,” New Delhi, India, 2011."]],
  scotland2021: [["Period Products (Free Provision) (Scotland) Act 2021, asp 1, Scottish Parliament, Edinburgh, U.K., 2021."]],
};

const refOrder = [];
function refNum(key) {
  if (!REFS[key]) throw new Error(`unknown reference ${key}`);
  let i = refOrder.indexOf(key);
  if (i === -1) { refOrder.push(key); i = refOrder.length - 1; }
  return i + 1;
}
// cite("a", "b") -> "[1], [2]"; consecutive runs of 3+ collapse to "[1]–[3]".
function cite(...keys) {
  const nums = keys.map(refNum).sort((a, b) => a - b);
  const parts = [];
  for (let i = 0; i < nums.length;) {
    let j = i;
    while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    if (j - i >= 2) parts.push(`[${nums[i]}]–[${nums[j]}]`);
    else for (let k = i; k <= j; k++) parts.push(`[${nums[k]}]`);
    i = j + 1;
  }
  return parts.join(", ");
}

// ---------------------------------------------------------------------------
// Inline markup: *italic*, ^superscript^, ~subscript~.
// ---------------------------------------------------------------------------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*[^*]+\*|\^[^^]+\^|~[^~]+~)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    const inner = tok.slice(1, -1);
    if (tok[0] === "*") out.push(new TextRun({ text: inner, ...base, italics: true }));
    else if (tok[0] === "^") out.push(new TextRun({ text: inner, ...base, superScript: true }));
    else out.push(new TextRun({ text: inner, ...base, subScript: true }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

// ---------------------------------------------------------------------------
// Paragraph builders, one per element type in the template.
// ---------------------------------------------------------------------------
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
let secNo = 0;
let subNo = 0;
let figNo = 0;
let tabNo = 0;

function body(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 288 },
    children: runs(text),
  });
}

// "1) Lead-in:" italic run followed by body text, as in the template.
function item(lead, text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 288 },
    children: [new TextRun({ text: lead, italics: true }), ...runs(text)],
  });
}

function heading(title, { numbered = true } = {}) {
  subNo = 0;
  const label = numbered ? `${ROMAN[secNo++]}.  ${title}` : title;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 80 },
    keepNext: true,
    children: [new TextRun({ text: label, smallCaps: true })],
  });
}

function subheading(title) {
  const letter = String.fromCharCode(65 + subNo++);
  return new Paragraph({
    spacing: { before: 80, after: 40 },
    keepNext: true,
    tabStops: [{ type: TabStopType.LEFT, position: 288 }],
    children: [new TextRun({ italics: true, children: [`${letter}.`, new Tab(), title] })],
  });
}

function figure(file, caption) {
  figNo++;
  const png = fs.readFileSync(path.join(HERE, "figures", file));
  const w = png.readUInt32BE(16);
  const h = png.readUInt32BE(20);
  const width = 318; // px at 96 dpi = 3.3 in, inside the 3.5-in column
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
      keepNext: true,
      children: [new ImageRun({
        type: "png", data: png,
        transformation: { width, height: Math.round(width * h / w) },
        altText: { title: `Fig. ${figNo}`, description: caption, name: file },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 160 },
      children: runs(`Fig. ${figNo}  ${caption}`, { size: 16 }),
    }),
  ];
}

const CELL_BORDER = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const ALL_BORDERS = { top: CELL_BORDER, bottom: CELL_BORDER, left: CELL_BORDER, right: CELL_BORDER };

function cell(text, width, { header = false, center = false } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: ALL_BORDERS,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 20, bottom: 20, left: 72, right: 72 },
    children: [new Paragraph({
      alignment: header || center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: runs(text, { size: 16, italics: header || undefined }),
    })],
  });
}

function table(title, widths, header, rows, note) {
  tabNo++;
  const total = widths.reduce((a, b) => a + b, 0);
  const out = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160 },
      keepNext: true,
      children: [new TextRun({ text: `TABLE ${ROMAN[tabNo - 1]}`, size: 16 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      keepNext: true,
      children: [new TextRun({ text: title, smallCaps: true })],
    }),
    new Table({
      width: { size: total, type: WidthType.DXA },
      columnWidths: widths,
      layout: TableLayoutType.FIXED,
      alignment: AlignmentType.CENTER,
      rows: [
        new TableRow({
          tableHeader: true,
          cantSplit: true,
          children: header.map((t, i) => cell(t, widths[i], { header: true })),
        }),
        ...rows.map((r) => new TableRow({
          cantSplit: true,
          children: r.map((t, i) => cell(t, widths[i], { center: i > 0 && widths[i] <= 1400 })),
        })),
      ],
    }),
  ];
  out.push(new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 40, after: 160 },
    children: note ? runs(note, { size: 16 }) : [],
  }));
  return out;
}

// Display equation: centred expression, number flush right in the column.
function equation(parts, n) {
  const children = [new TextRun({ children: [new Tab()] })];
  for (const [t, f = ""] of parts) {
    children.push(new TextRun({
      text: t,
      italics: f.includes("i"),
      subScript: f.includes("s"),
      font: f.includes("m") ? "Cambria Math" : FONT,
    }));
  }
  children.push(new TextRun({ children: [new Tab(), `(${n})`] }));
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    tabStops: [
      { type: TabStopType.CENTER, position: COL_W / 2 },
      { type: TabStopType.RIGHT, position: COL_W },
    ],
    children,
  });
}

function spacer(size = 12) {
  return new Paragraph({ children: [new TextRun({ text: "", size })] });
}

// ---------------------------------------------------------------------------
// Title and author block (single column).
// ---------------------------------------------------------------------------
const TITLE = "Menstrual Health and Hygiene: Practices, Determinants and Evidence-Based Interventions";

const NO_BORDER = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NO_BORDERS = { top: NO_BORDER, bottom: NO_BORDER, left: NO_BORDER, right: NO_BORDER };

function authorCell(lines) {
  const [name, dept, inst, place, email] = lines;
  const line = (text, opts = {}, spacing = {}) => new Paragraph({
    alignment: AlignmentType.CENTER, spacing, children: [new TextRun({ text, ...opts })],
  });
  return new TableCell({
    width: { size: FULL_W / 2, type: WidthType.DXA },
    borders: NO_BORDERS,
    verticalAlign: VerticalAlign.CENTER,
    children: [
      line(name, { size: 22 }, { after: 60 }),
      line(dept, { italics: true }),
      line(inst, { italics: true }),
      line(place, { italics: true }),
      line(email, {}, { before: 60 }),
    ],
  });
}

const titleBlock = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({ text: TITLE, size: 44 }),
      new FootnoteReferenceRun(1),
    ],
  }),
  spacer(12),
  new Table({
    width: { size: FULL_W, type: WidthType.DXA },
    columnWidths: [FULL_W / 2, FULL_W / 2],
    layout: TableLayoutType.FIXED,
    borders: { ...NO_BORDERS, insideHorizontal: NO_BORDER, insideVertical: NO_BORDER },
    rows: [new TableRow({
      cantSplit: true,
      children: [
        authorCell(["[First Author] and [Second Author]", "[Department of Public Health]",
          "[Name of University]", "[City, State, Country]", "{first.author, second.author}@university.edu"]),
        authorCell(["[Third Author]", "[Department of Biomedical Engineering]",
          "[Name of University]", "[City, State, Country]", "third.author@university.edu"]),
      ],
    })],
  }),
  spacer(18),
];

// ---------------------------------------------------------------------------
// Main text (two columns). Built in reading order so citations number in order.
// ---------------------------------------------------------------------------
const main = [];
const add = (...xs) => xs.flat().forEach((x) => main.push(x));

add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: 288 },
  children: [
    new TextRun({ text: "Abstract", bold: true, italics: true, size: 18 }),
    new TextRun({
      bold: true, size: 18,
      text: " - Menstruation is a normal biological process experienced every month by about 1.8 billion people, yet inadequate access to menstrual materials, water, sanitation and hygiene (WASH), accurate information and supportive social environments continues to undermine health, education and dignity. This paper reviews the current state of menstrual health and hygiene (MHH) and analyses publicly available secondary data from the WHO/UNICEF Joint Monitoring Programme (JMP) and two rounds of India’s National Family Health Survey (NFHS-4, 2015–16, and NFHS-5, 2019–21). Globally, one in four people lacked a basic handwashing facility at home in 2022, and in 2023 only 39% of schools provided menstrual health education and 31% had bins for menstrual waste. In India, the share of women aged 15–24 using hygienic methods of menstrual protection rose from 57.6% to 77.3%, but a 16.8-percentage-point urban–rural gap persists and use is strongly graded by schooling and household wealth. A simple life-cycle model indicates that one user of disposable products discards close to 10,000 units over a lifetime, compared with four menstrual cups. We synthesize evidence on health, psychosocial, educational and environmental outcomes, map barriers onto a socio-ecological framework and propose integrated recommendations spanning education, product access, school infrastructure, health services, waste management and policy.",
    }),
  ],
}));
add(spacer(12));
add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: 288 },
  children: [new TextRun({
    text: "Index Terms - Menstrual health, menstrual hygiene management, period poverty, WASH, adolescent health.",
    bold: true, italics: true, size: 18,
  })],
}));
add(spacer(12));

// I. Introduction
add(heading("Introduction"));
add(body(`Menstruation is a normal and healthy biological process. Every month about 1.8 billion girls, women, transgender men and non-binary persons menstruate, and on any given day more than 300 million people are menstruating worldwide ${cite("worldbank", "unicef2019")}. A person may experience 450 to 500 menstrual periods over a reproductive lifetime. Managing them safely and with dignity requires far more than an absorbent product: it requires accurate information, clean water and soap, private and safe sanitation facilities, a way to dispose of used materials, access to care for menstrual disorders, and a social environment free of stigma ${cite("hennegan2021", "sommer2015")}.`));
add(body(`When these conditions are not met, menstruation becomes a source of disadvantage. The term *period poverty* describes the inability to afford menstrual products and related WASH services, and it is not confined to low-income countries: a 2017 Plan International UK survey reported that one in ten girls aged 14–21 in the United Kingdom had been unable to afford sanitary products ${cite("plan2017")}. In low- and middle-income countries (LMICs), qualitative syntheses document shame, fear of leakage, restrictions on mobility and diet, and missed school or work ${cite("hennegan2019")}. These experiences undermine progress on several Sustainable Development Goals (SDGs), including good health (SDG 3), quality education (SDG 4), gender equality (SDG 5) and clean water and sanitation (SDG 6), whose Target 6.2 explicitly calls for “special attention to the needs of women and girls” ${cite("un2015")}.`));

add(subheading("Objectives and Contributions"));
add(body("This paper consolidates current knowledge on menstrual health and hygiene (MHH) and quantifies its status using recent, publicly available data. Its specific contributions are as follows."));
add(item("1) Conceptual synthesis", ": We trace the shift from menstrual hygiene management (MHM) to the broader concept of menstrual health and summarize the components and indicators now used for monitoring."));
add(item("2) Secondary data analysis", ": We analyse global estimates from the WHO/UNICEF Joint Monitoring Programme (JMP) and two rounds of India’s National Family Health Survey (NFHS) to describe levels, trends and inequalities in MHH."));
add(item("3) Environmental estimate", ": We introduce a transparent life-cycle model of the number of menstrual products used per person to compare disposable and reusable options."));
add(item("4) Recommendations", ": We map barriers onto a socio-ecological framework and propose evidence-informed actions for policy and practice."));
add(body("The remainder of the paper is organized as follows. Section II defines key concepts, Section III describes the methodology, Section IV presents the results, Section V discusses barriers, interventions and limitations, and Section VI concludes."));

// II. Background
add(heading("Background and Definitions"));
add(subheading("From Menstrual Hygiene Management to Menstrual Health"));
add(body(`In 2012 the JMP proposed a definition of MHM in which women and adolescent girls use “a clean menstrual management material to absorb or collect menstrual blood, that can be changed in privacy as often as necessary for the duration of the menstrual period, using soap and water for washing the body as required, and having access to safe and convenient facilities to dispose of used menstrual management materials” ${cite("jmp2012")}. The definition brought menstruation onto the WASH agenda, but it focuses largely on materials and infrastructure.`));
add(body(`Hennegan *et al.* ${cite("hennegan2021")} subsequently defined menstrual health as “a state of complete physical, mental, and social well-being and not merely the absence of disease or infirmity, in relation to the menstrual cycle.” Achieving it requires the five components summarized in Table I. This broader framing recognizes that knowledge, pain management, social norms and participation matter as much as products and toilets ${cite("unicef2019")}.`));
add(table(
  "Components of Menstrual Health",
  [1250, 3700],
  ["Component", "Requirement"],
  [
    ["Information", "Accurate, timely, age-appropriate information about the menstrual cycle, menstruation and self-care"],
    ["Care", "Effective, affordable materials and supportive WASH and disposal facilities, used in comfort and privacy"],
    ["Health care", "Timely diagnosis, treatment and care for menstrual discomfort and disorders"],
    ["Environment", "A positive, respectful environment free from stigma and psychological distress"],
    ["Participation", "Freedom to take part in civil, cultural, economic, social and political life during all phases of the cycle"],
  ],
  `Adapted from ${cite("hennegan2021")}.`,
));

add(subheading("Physiology and Common Disorders"));
add(body(`The menstrual cycle is regulated by the hypothalamic–pituitary–ovarian axis. Follicle-stimulating hormone and luteinizing hormone drive follicular development and ovulation, while ovarian estrogen and progesterone prepare the endometrium; if pregnancy does not occur, the fall in progesterone triggers shedding of the endometrium as menstrual bleeding. Cycles of roughly 24–38 days with bleeding lasting up to eight days fall within internationally accepted normal limits ${cite("munro2018")}.`));
add(body(`Menstrual disorders are common and often under-treated. Primary dysmenorrhea affects an estimated 45%–95% of menstruating women ${cite("iacovides2015")} and is a frequent cause of absence from school and work. Heavy menstrual bleeding contributes to iron-deficiency anaemia, while endometriosis, which affects roughly 10% of women and girls of reproductive age, and polycystic ovary syndrome, which affects an estimated 8%–13%, frequently go undiagnosed for years ${cite("whoendo", "whopcos")}. Breaking the silence around menstruation is therefore also a prerequisite for timely care.`));

add(subheading("Monitoring Menstrual Health"));
add(body(`Global monitoring has historically relied on proxies such as household access to handwashing facilities and sanitation. The JMP now reports MHH indicators for households and schools, including awareness of menstruation before menarche, use of menstrual materials, access to a private place to wash and change, exclusion from activities during menstruation and, for schools, the provision of menstrual health education and of bins for menstrual waste ${cite("jmp2023", "jmp2024")}. Nationally, India’s NFHS has asked women aged 15–24 since 2015–16 which methods of protection they use during menstruation, providing one of the largest population-based time series on the topic ${cite("nfhs4", "nfhs5")}.`));

// III. Methodology
add(heading("Methodology"));
add(subheading("Study Design"));
add(body("This study combines a narrative review of the scientific and policy literature with a descriptive analysis of secondary, publicly available data. No primary data were collected and no human participants were involved; ethical approval was therefore not required."));
add(subheading("Data Sources"));
add(item("1) Global household data", `: The JMP 2023 report ${cite("jmp2023")} provides 2022 estimates of the hygiene service ladder, defined by the availability of a handwashing facility with soap and water at home.`));
add(item("2) Global school data", `: The JMP 2024 schools report ${cite("jmp2024")} provides 2023 estimates of basic drinking water, sanitation and hygiene services in schools, together with the first global estimates of menstrual health provision in schools, which are based on countries with available data.`));
add(item("3) National data for India", `: NFHS-4 (2015–16) and NFHS-5 (2019–21) are nationally representative household surveys conducted by the International Institute for Population Sciences ${cite("nfhs4", "nfhs5")}; NFHS-5 covered more than 600,000 households. Women aged 15–24 were asked which methods of protection they use during menstruation. Locally prepared napkins, sanitary napkins, tampons and menstrual cups are classified as hygienic methods.`));
add(item("4) Literature", ": Peer-reviewed articles were identified through PubMed and Google Scholar using combinations of the terms “menstrual health”, “menstrual hygiene”, “period poverty”, “menstrual cup”, “school absenteeism” and “reproductive tract infection”. Systematic reviews, meta-analyses and controlled trials were prioritized and supplemented by reports from United Nations agencies and national governments."));
add(subheading("Analytical Approach"));
add(body("Indicators are reported as percentages. Change between survey rounds is expressed in percentage points (pp), and inequality is expressed both as an absolute gap between two groups and as a relative ratio. To compare menstrual products, we estimate the number of units one person uses over a menstruating lifetime. For single-use products,"));
add(equation([["N", "i"], ["d", "is"], [" = "], ["c", "i"], [" · "], ["Y", "i"], [" · "], ["d", "i"], [" · "], ["p", "i"]], 1));
add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  children: runs("where *c* is the number of cycles per year, *Y* the number of menstruating years, *d* the number of bleeding days per cycle and *p* the number of products used per day. For reusable products,"),
}));
add(equation([["N", "i"], ["r", "is"], [" = "], ["k", "i"], [" · "], ["⌈", "m"], ["Y", "i"], [" / "], ["L", "i"], ["⌉", "m"]], 2));
add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  children: [
    ...runs("where *k* is the number of units in one set, *L* the useful life of the set in years and "),
    new TextRun({ text: "⌈·⌉", font: "Cambria Math" }),
    ...runs(" denotes rounding up to a whole number. Parameter values are listed in Table III; they are typical rather than exact values and are intended to show orders of magnitude."),
  ],
}));

// IV. Results
add(heading("Results"));
add(subheading("Global Status of Hygiene and School WASH"));
add(body(`In 2022, 75% of the world’s population had a basic hygiene service at home, 17% had a facility lacking water or soap and 8% had no handwashing facility at all (Fig. 1). In absolute terms, 2.0 billion people lacked basic hygiene services, including 653 million with no facility ${cite("jmp2023")}. Because washing the hands and body with soap and water is integral to safe menstrual care, particularly for users of reusable materials, these gaps translate directly into barriers to menstrual hygiene.`));
add(figure("fig1_hygiene_ladder_pie.png", `Global household hygiene service levels, 2022 (data: WHO/UNICEF JMP ${cite("jmp2023")}).`));
add(body(`The situation in schools, where adolescent girls spend much of the day, is shown in Fig. 2. In 2023, 77% of schools worldwide had a basic drinking water service, 78% basic sanitation and only 67% basic hygiene, so one school in three lacked handwashing with soap and water ${cite("jmp2024")}. Provision specific to menstruation lagged further behind: only 39% of schools provided menstrual health education and 31% had bins for menstrual waste in girls’ toilets. A girl attending such a school may have nowhere to wash, change or discreetly dispose of used materials during the school day.`));
add(figure("fig2_school_wash_bar.png", `Basic WASH services and menstrual health provision in schools worldwide, 2023 (data: WHO/UNICEF JMP ${cite("jmp2024")}).`));

add(subheading("Trends in Hygienic Method Use in India"));
add(body(`Fig. 3 compares the two most recent NFHS rounds. Nationally, the share of women aged 15–24 using a hygienic method rose from 57.6% in 2015–16 to 77.3% in 2019–21, an increase of 19.7 pp in about five years ${cite("nfhs4", "nfhs5")}. The gain was largest in rural areas (+24.4 pp, from 48.2% to 72.6%) compared with urban areas (+11.9 pp, from 77.5% to 89.4%), so the urban–rural gap narrowed from 29.3 pp to 16.8 pp.`));
add(figure("fig3_nfhs_trend_bar.png", `Women aged 15–24 in India using hygienic methods of menstrual protection, NFHS-4 vs. NFHS-5 (data: ${cite("nfhs4", "nfhs5")}).`));
add(body("Despite this progress, a substantial share of young women still use no hygienic method at all. As Fig. 4 shows, 10.6% of urban and 27.4% of rural women aged 15–24 reported no hygienic method in 2019–21; given the size of India’s youth population, this corresponds to many millions of young women."));
add(figure("fig4_urban_rural_pie.png", `Share of women aged 15–24 using a hygienic method in urban and rural India, NFHS-5 (data: ${cite("nfhs5")}).`));

add(subheading("Methods of Menstrual Protection"));
add(body(`The methods reported in NFHS-5 are shown in Fig. 5. Because respondents could report more than one method, the percentages sum to more than 100. Commercial sanitary napkins were the most common (64.4%), but nearly half of young women (49.6%) used cloth, often alongside other methods, and 15.0% used locally prepared napkins. Insertable products remained rare: 1.7% reported tampons and only 0.3% reported menstrual cups ${cite("nfhs5")}. NFHS classifies cloth as non-hygienic regardless of how it is cared for, yet cloth that is washed with soap, dried in sunlight and stored cleanly can be a safe and sustainable option; the indicator therefore measures the type of material rather than the quality of practice.`));
add(figure("fig5_methods_bar.png", `Methods of menstrual protection used by women aged 15–24 in India, NFHS-5; multiple responses allowed (data: ${cite("nfhs5")}).`));

add(subheading("Socio-economic Determinants"));
add(body(`Use of hygienic methods is steeply graded by education and wealth (Fig. 6). Women with 12 or more years of schooling were more than twice as likely to use a hygienic method as women with no schooling (90% vs. 44%), and women in the highest wealth quintile were almost twice as likely as those in the lowest (95% vs. 54%) ${cite("nfhs5")}. These gradients are consistent with a meta-analysis of 138 Indian studies, which found that commercial pad use was more common among urban (67%) than rural (32%) adolescent girls and that only about half of girls (48%) had been informed about menstruation before menarche ${cite("vaneijk2016")}. Affordability, knowledge and local availability therefore act together.`));
add(figure("fig6_determinants_bar.png", `Use of hygienic methods among women aged 15–24 by schooling and household wealth, India, NFHS-5 (data: ${cite("nfhs5")}).`));

add(subheading("Health, Psychosocial and Educational Outcomes"));
add(item("1) Reproductive and urinary tract infections", `: A study of women in Odisha, India, found that those using reusable absorbent pads were more likely to report symptoms of urogenital infection or to be diagnosed with an infection than women using disposable pads ${cite("das2015")}. However, a systematic review concluded that the overall evidence linking menstrual practices to infection is limited and heterogeneous ${cite("sumpter2013")}, and causal pathways such as inadequate washing and drying of reusable materials require further study.`));
add(item("2) Pain and psychosocial well-being", `: Dysmenorrhea is among the most common gynaecological complaints ${cite("iacovides2015")}, yet many adolescents receive no advice on pain relief. A qualitative metasynthesis of studies from LMICs described menstruation as a time of shame, fear of leakage and social exclusion, with girls adopting concealment strategies that cause distress ${cite("hennegan2019")}.`));
add(item("3) Education", `: In India, about one in four adolescent girls (24%) reported missing school during menstruation ${cite("vaneijk2016")}. Absence is driven by a combination of pain, fear of staining, lack of materials and inadequate school toilets, which links the school-level gaps in Fig. 2 directly to educational outcomes.`));

add(subheading("Environmental Impact of Menstrual Products"));
add(body(`The shift towards commercial disposable napkins, while improving hygiene, has created a waste-management challenge. An estimated 12.3 billion used sanitary napkins, about 113,000 tonnes, reach Indian landfills every year, and conventional napkins can be made of up to 90% plastic ${cite("toxics2021")}. Used products are frequently mixed with household waste, exposing sanitation workers to unhygienic conditions.`));
add(body(`Table II compares common menstrual products, and Fig. 7 applies (1) and (2) with the parameters in Table III. A single person using disposable pads or tampons may discard about 9,880 units over a menstruating lifetime. Dividing the reported waste mass by the number of napkins in ${cite("toxics2021")} gives roughly 9 g per napkin, implying about 90 kg of waste per person. In contrast, the same period could be covered by about 228 reusable cloth pads, 133 pieces of period underwear or four menstrual cups. A systematic review and meta-analysis found that leakage with menstrual cups was similar to or lower than with disposable pads or tampons, with no evidence of increased infection risk ${cite("vaneijk2019")}, although users need clean water, privacy and initial guidance.`));
add(table(
  "Comparison of Common Menstrual Products",
  [1250, 900, 950, 1850],
  ["Product", "Reusable", "Typical wear", "Main considerations"],
  [
    ["Disposable pad", "No", "4–8 h", "Widely available; plastic waste; recurring cost"],
    ["Tampon", "No", "Up to 8 h", "Low uptake in India; rare risk of toxic shock syndrome"],
    ["Cloth or cloth pad", "Yes", "4–6 h", "Low cost; needs soap, water and sun-drying"],
    ["Menstrual cup", "Yes, 5–10 yr", "Up to 12 h", "Least waste; needs clean water and initial training"],
    ["Period underwear", "Yes, ~2 yr", "Up to 12 h", "Higher upfront cost; suitability depends on flow"],
  ],
  "Wear times are typical manufacturer or public-health guidance and vary with flow.",
));
add(table(
  "Parameters Used in (1) and (2)",
  [2650, 900, 1400],
  ["Parameter", "Symbol", "Value"],
  [
    ["Cycles per year", "*c*", "13"],
    ["Menstruating years", "*Y*", "38"],
    ["Bleeding days per cycle", "*d*", "5"],
    ["Disposable products per day", "*p*", "4"],
    ["Cloth pads: set size; life", "*k*; *L*", "12; 2 yr"],
    ["Period underwear: set size; life", "*k*; *L*", "7; 2 yr"],
    ["Menstrual cup: set size; life", "*k*; *L*", "1; 10 yr"],
  ],
  "Typical values chosen to illustrate orders of magnitude.",
));
add(figure("fig7_lifetime_units_bar.png", "Estimated number of menstrual products used by one person over a menstruating lifetime, computed from (1), (2) and Table III (log scale)."));

// V. Discussion
add(heading("Discussion"));
add(subheading("A Socio-ecological View of Barriers"));
add(body(`The results show that MHH outcomes are shaped by factors operating at several levels at once. Table IV organizes the main barriers using a socio-ecological framework, in line with the multi-level perspectives used in the MHH literature ${cite("sommer2015", "hennegan2019")}, and pairs each level with example interventions. The steep gradients by schooling and wealth (Fig. 6) indicate that individual-level barriers such as knowledge and affordability remain decisive, while the school data (Fig. 2) show that environmental barriers persist even where products are available.`));
add(table(
  "Barriers and Interventions by Level",
  [1100, 1950, 1900],
  ["Level", "Key barriers", "Example interventions"],
  [
    ["Individual", "Little knowledge before menarche; cost of products; pain", "Puberty and MHH education; subsidized or free products; pain management"],
    ["Interpersonal", "Silence within families; teasing; restrictive norms", "Engage parents, teachers, boys and men; peer educators"],
    ["Institutional", "Toilets without water, soap, locks or bins; no products at school", "School WASH upgrades; emergency product supply; disposal systems"],
    ["Societal and policy", "Stigma; taxation; weak waste regulation; data gaps", "Tax removal; free-provision laws; product standards; routine monitoring"],
  ],
  "",
));
add(subheading("Evidence from Interventions"));
add(body(`Controlled trials of product provision show promising but mixed effects. In a cluster randomized feasibility study in rural Western Kenya, providing menstrual cups or monthly sanitary pads did not reduce school dropout over roughly one year of follow-up, although the study reported lower prevalence of sexually transmitted infections in the intervention arms and good acceptability of both products ${cite("phillips2016")}. In Uganda, a quasi-randomized trial in eight schools reported that providing reusable sanitary pads or puberty education was associated with improved school attendance ${cite("montgomery2016")}. A systematic review of MHM interventions in LMICs judged the evidence for educational and psychosocial benefits to be promising but limited by study quality ${cite("hennegan2016")}, underscoring the need for rigorous, adequately powered trials.`));
add(body(`Policy levers are increasingly being used. India launched a scheme in 2011 to provide subsidized sanitary napkins to adolescent girls in rural areas ${cite("mohfw2011")}, exempted sanitary napkins from the Goods and Services Tax in 2018 and supplies low-cost napkins through public pharmacies. Scotland became the first country to legislate universal free access to period products through the Period Products (Free Provision) (Scotland) Act 2021 ${cite("scotland2021")}. The improvements observed between NFHS-4 and NFHS-5 coincide with such efforts, although survey data alone cannot establish causality.`));
add(subheading("Recommendations"));
add(body("Based on the evidence reviewed, we propose six priority actions."));
add(item("1) Education before menarche", ": Integrate comprehensive, age-appropriate MHH education into school curricula for all genders, delivered before girls reach menarche."));
add(item("2) MHH-ready schools", ": Ensure every school has functional, private, gender-segregated toilets with water, soap, locks and bins for menstrual waste, together with an emergency supply of products."));
add(item("3) Affordable choice", ": Provide free or subsidized products, including reusable options such as cloth pads and menstrual cups, with guidance on their safe use and care."));
add(item("4) Sustainable waste management", ": Promote source segregation of menstrual waste, set standards for genuinely compostable products and apply extended producer responsibility to manufacturers of disposable products."));
add(item("5) Health services", ": Train frontline health workers to recognize and manage dysmenorrhea, heavy bleeding and conditions such as endometriosis and polycystic ovary syndrome."));
add(item("6) Inclusive data", ": Adopt the JMP indicators in national surveys and disaggregate them by disability, geography and socio-economic status, so that marginalized groups, including people with disabilities, displaced populations and transgender and non-binary people who menstruate, are visible to policy."));
add(subheading("Limitations"));
add(body("This study has several limitations. First, it relies on cross-sectional, self-reported survey data that may be affected by social desirability bias. Second, the NFHS indicator classifies methods by material type and does not capture washing, drying, frequency of change or disposal practices. Third, the global school estimates of menstrual health provision are based on the subset of countries with available data. Fourth, the life-cycle model uses simplified, typical parameters and ignores variation in cycle length, flow, pregnancy and product mass, so its results should be read as orders of magnitude. Finally, as a narrative rather than a systematic review, the literature synthesis may be subject to selection bias."));

// VI. Conclusion
add(heading("Conclusion"));
add(body("Menstrual health is a matter of health, education, gender equality and environmental sustainability. Global data show that one in four people lack basic hygiene at home and that most schools still provide neither menstrual health education nor bins for menstrual waste. India’s experience demonstrates that rapid progress is possible, with hygienic method use among young women rising by nearly 20 pp in about five years, yet large inequalities by residence, schooling and wealth remain. At the same time, growing reliance on disposable products is generating a substantial plastic-waste burden that reusable alternatives could cut from roughly 40-fold to more than 2,000-fold in units per user. Integrated strategies that combine education, affordable product choice, MHH-ready schools, health services, waste management and better data are needed so that everyone who menstruates can do so safely, comfortably and with dignity."));

// Acknowledgment and references
add(heading("Acknowledgment", { numbered: false }));
add(body("The authors thank the International Institute for Population Sciences, Mumbai, and the WHO/UNICEF Joint Monitoring Programme for making the survey data used in this study publicly available."));
add(heading("References", { numbered: false }));
for (const key of refOrder) {
  const hasUrl = REFS[key].some(([t]) => t.includes("http"));
  add(new Paragraph({
    alignment: hasUrl ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
    numbering: { reference: "refs", level: 0 },
    children: REFS[key].map(([t, f]) => new TextRun({ text: t, size: 16, italics: f === "i" })),
  }));
}
const unused = Object.keys(REFS).filter((k) => !refOrder.includes(k));
if (unused.length) throw new Error(`uncited references: ${unused.join(", ")}`);

// ---------------------------------------------------------------------------
// Document assembly.
// ---------------------------------------------------------------------------
const pageProps = { page: { size: PAGE, margin: MARGIN } };

const doc = new Document({
  creator: "Author",
  title: TITLE,
  description: "Research paper on menstrual health and hygiene in IEEE conference format",
  styles: {
    default: {
      document: { run: { font: FONT, size: 20 } },
    },
  },
  numbering: {
    config: [{
      reference: "refs",
      levels: [{
        level: 0,
        format: LevelFormat.DECIMAL,
        text: "[%1]",
        alignment: AlignmentType.LEFT,
        style: {
          paragraph: { indent: { left: 360, hanging: 360 } },
          run: { size: 16 },
        },
      }],
    }],
  },
  footnotes: {
    1: { children: [new Paragraph({ children: [new TextRun({ text: " This research received no specific grant from any funding agency. [Replace with funding details if applicable.]", size: 16 })] })] },
  },
  sections: [
    { properties: { ...pageProps }, children: titleBlock },
    {
      properties: {
        ...pageProps,
        type: SectionType.CONTINUOUS,
        column: { count: 2, space: COL_GAP, equalWidth: true },
      },
      children: main,
    },
    // An empty continuous section after the text makes Word balance the two
    // columns on the last page, as the IEEE template asks.
    {
      properties: {
        ...pageProps,
        type: SectionType.CONTINUOUS,
        column: { count: 2, space: COL_GAP, equalWidth: true },
      },
      children: [new Paragraph({ children: [] })],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  execFileSync("python3", [path.join(HERE, "finalize_docx.py"), OUT], { stdio: "inherit" });
  console.log(`wrote ${OUT} (${refOrder.length} references, ${figNo} figures, ${tabNo} tables)`);
});
