import { useEffect, useState, useRef } from 'react'
import { Box, Container, IconButton, Switch, Image, Link } from 'theme-ui'
import { Search, X, ChevronUp, ChevronDown } from 'lucide-react'
import { Themed } from '@theme-ui/mdx'
import maplibregl from 'maplibre-gl'
import { scaleLinear } from 'd3-scale'
import { interpolateLab } from 'd3-interpolate'
import { range } from 'd3-array'
import {
  Row,
  Column,
  Input,
  Autocomplete,
  Results,
  Options,
  Colorbar,
} from '../components'
import { addLabels, addShapes } from '../components/layers'
import {
  handleShapeClick,
  handleShapeClickOff,
  handleShapeMouseMove,
  handleShapeMouseMoveoff,
  createShapeMouseMoveHandler,
  createShapeMouseMoveOffHandler,
  createShapeClickHandler,
  createShapeClickOffHandler,
  updateShapeVisibility,
  getMaxKey,
} from '../components/utils'
import {
  boroughColors,
  candidates,
  candidateColors,
  mapStyles,
  scaleLookup,
  raceLookup,
} from '../components/constants'
import load from '../components/load'
import shapes from '../data'

const Index = () => {
  const { data, error } = load()

  const map = useRef()
  const [selected, setSelected] = useState({})
  const [selectedCandidate, setSelectedCandidate] = useState('All Candidates')
  const [scale, setScale] = useState('Assembly district')
  const [race, setRace] = useState('Mayoral')
  const [candidateColorScales, setCandidateColorScales] = useState({})

  const setup = async () => {
    addShapes(map.current, 'election-districts', 0.25)
    addShapes(map.current, 'assembly-districts', 0.25)
    addLabels(map.current)
  }

  useEffect(() => {
    const candidateColorScales = {}
    candidates[race].forEach((d) => {
      candidateColorScales[d] = scaleLinear()
        .domain([0, 1])
        .interpolate(interpolateLab)
        .range(['white', candidateColors[d]])
    })

    setCandidateColorScales(candidateColorScales)
  }, [race])

  useEffect(() => {
    const isMobile =
      typeof window !== 'undefined' &&
      window.matchMedia('(max-width: 40em)').matches

    const defaultZoom = isMobile ? 9.8 : 10.4

    map.current = new maplibregl.Map({
      container: 'map',
      style: mapStyles.monochrome,
      center: [-73.956, 40.7228],
      zoom: defaultZoom,
      minZoom: 9,
    })

    map.current.on('load', () => {
      setup()
    })

    return () => map.current.remove()
  }, [])

  useEffect(() => {
    if (!map.current) return

    const layers = ['assembly-district', 'election-district']
    const handlers = {}
    const foo = 2

    layers.forEach((layer) => {
      handlers[layer] = {}
      handlers[layer].move = createShapeMouseMoveHandler(map.current, layer)
      handlers[layer].moveOff = createShapeMouseMoveOffHandler(map.current)
      handlers[layer].click = createShapeClickHandler(
        map.current,
        layer,
        setSelected,
      )
      handlers[layer].clickOff = createShapeClickOffHandler(
        map.current,
        layer,
        setSelected,
      )

      const addHandlers = () => {
        map.current.on('mousemove', `${layer}s-fill`, handlers[layer].move)
        map.current.on('click', `${layer}s-fill`, handlers[layer].click)
        map.current.on('mousemove', handlers[layer].moveOff)
        map.current.on('click', handlers[layer].clickOff)
      }

      if (!map.current.isStyleLoaded()) {
        map.current.once('idle', () => {
          addHandlers()
        })
      } else {
        addHandlers()
      }
    })

    return () => {
      layers.forEach((layer) => {
        map.current.off('mousemove', `${layer}s-fill`, handlers[layer].move)
        map.current.off('click', `${layer}s-fill`, handlers[layer].click)
        map.current.off('mousemove', handlers[layer].moveOff)
        map.current.off('click', handlers[layer].clickOff)
      })
    }
  }, [map.current, scale])

  useEffect(() => {
    if (!map.current) return

    const update = () => {
      const layers = ['election-district', 'assembly-district']

      updateShapeVisibility(map.current, `${scaleLookup[scale]}s`, 'visible')

      layers.forEach((l) => {
        if (!(l == scaleLookup[scale])) {
          updateShapeVisibility(map.current, `${l}s`, 'none')
        }
      })
    }

    if (!map.current.isStyleLoaded()) {
      map.current.once('idle', () => {
        update()
      })
    } else {
      update()
    }
  }, [map.current, scale])

  useEffect(() => {
    if (!candidates[race].includes(selectedCandidate)) {
      setSelectedCandidate('All Candidates')
    }
  }, [race])

  useEffect(() => {
    const updateSelected = () => {
      if (scale == 'Election district') {
        Object.keys(shapes['election-districts']).forEach((k) => {
          const districtResults = data[raceLookup[race]].election_districts[k]
          let color = '#cccccc'
          let opacity = 0

          if (districtResults) {
            if (selectedCandidate === 'All Candidates') {
              // Show leading candidate for each district
              const leadingCandidate = getMaxKey(districtResults.candidates)
              if (leadingCandidate) {
                color = candidateColors[leadingCandidate] || '#cccccc'
                opacity = districtResults.reporting > 0 ? 1 : 0 // Scale opacity based on reporting
              }
            } else if (
              districtResults.candidates[selectedCandidate] !== undefined
            ) {
              // Show only the selected candidate's results
              const totalVotes = districtResults.total
              const candidateVoteShare =
                totalVotes > 0
                  ? districtResults.candidates[selectedCandidate] / totalVotes
                  : 0

              // Show the candidate's color with opacity based on vote share and reporting
              color =
                candidateColorScales[selectedCandidate](candidateVoteShare)
              opacity = 1
            }
            // If candidate not in district, keep opacity at 0 (invisible)
            map.current.setFeatureState(
              {
                source: 'election-districts',
                id: shapes['election-districts'][k].id,
              },
              {
                color,
                opacity,
                'line-width': selected[scaleLookup[scale]] === k ? 1.5 : 0.25,
              },
            )
          } else {
            map.current.setFeatureState(
              {
                source: 'election-districts',
                id: shapes['election-districts'][k].id,
              },
              {
                color: '#cccccc',
                opacity: 0,
                'line-width': selected[scaleLookup[scale]] === k ? 1.5 : 0.25,
              },
            )
          }
        })
      }

      if (scale == 'Assembly district') {
        Object.keys(shapes['assembly-districts']).forEach((k) => {
          const districtResults = data[raceLookup[race]].assembly_districts[k]
          let color = '#cccccc'
          let opacity = 0

          if (districtResults) {
            if (selectedCandidate === 'All Candidates') {
              // Show leading candidate for each district
              const leadingCandidate = getMaxKey(districtResults.candidates)
              if (leadingCandidate) {
                color = candidateColors[leadingCandidate] || '#cccccc'
                opacity = districtResults.reporting > 0 ? 1 : 0 // Scale opacity based on reporting
              }
            } else if (
              districtResults.candidates[selectedCandidate] !== undefined
            ) {
              // Show only the selected candidate's results
              const totalVotes = districtResults.total
              const candidateVoteShare =
                totalVotes > 0
                  ? districtResults.candidates[selectedCandidate] / totalVotes
                  : 0

              // Show the candidate's color with opacity based on vote share and reporting
              color =
                candidateColorScales[selectedCandidate](candidateVoteShare)
              opacity = 1
            }
            // If candidate not in district, keep opacity at 0 (invisible)
            map.current.setFeatureState(
              {
                source: 'assembly-districts',
                id: shapes['assembly-districts'][k].id,
              },
              {
                color,
                opacity,
                'line-width': selected[scaleLookup[scale]] === k ? 1.5 : 0.5,
              },
            )
          } else {
            map.current.setFeatureState(
              {
                source: 'assembly-districts',
                id: shapes['assembly-districts'][k].id,
              },
              {
                color: '#cccccc',
                opacity: 0,
                'line-width': selected[scaleLookup[scale]] === k ? 1.5 : 0.5,
              },
            )
          }
        })
      }
    }

    if (data[raceLookup[race]]) {
      if (map.current) {
        if (!map.current.isStyleLoaded()) {
          map.current.once('idle', () => {
            updateSelected()
          })
        } else {
          updateSelected()
        }
      }
    }
  }, [data, selected, selectedCandidate, scale, race])

  const resetView = () => {
    map.current.flyTo({
      center: [-73.956, 40.7228],
      zoom: 10.4,
    })
  }

  return (
    <>
      {data.status && data.status.error && (
        <Box sx={{ position: 'absolute', top: 0, left: 0, zIndex: 2000 }}>
          <Box
            sx={{ bg: 'rgb(255,255,255,0.9)', width: '100vw', height: '100vh' }}
          >
            <Box
              sx={{
                fontFamily: 'heading',
                lineHeight: '1.2em',
                letterSpacing: 'heading',
                fontSize: [5, 6, 6, 6],
                position: 'absolute',
                textAlign: 'center',
                left: '50%',
                top: '50%',
                transform: 'translate(-50%, -50%)',
              }}
            >
              Check in soon for results.
            </Box>
          </Box>
        </Box>
      )}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          zIndex: 1000,
          borderRadius: 2,
        }}
      >
        <Box
          sx={{
            borderRadius: [0, '2px', '2px', '2px'],
            bg: 'rgb(255,255,255,0.9)',
            ml: [0, 4, 4, 4],
            mt: [0, 4, 4, 4],
            width: ['calc(100vw)', '400px', '400px', '400px'],
          }}
        >
          <Results
            selected={selected}
            scale={scale}
            race={race}
            setSelectedCandidate={setSelectedCandidate}
          />
        </Box>
      </Box>
      <Box
        sx={{
          position: 'absolute',
          bottom: [0, 'initial', 'initial', 'initial'],
          top: ['initial', 0, 0, 0],
          right: 0,
        }}
      >
        <Options
          selectedCandidate={selectedCandidate}
          setSelectedCandidate={setSelectedCandidate}
          scale={scale}
          setScale={setScale}
          race={race}
          setRace={setRace}
        />
      </Box>
      {selectedCandidate && candidateColorScales[selectedCandidate] && (
        <Box
          sx={{
            position: 'absolute',
            right: ['24px'],
            top: ['380px'],
            zIndex: 5000,
            color: 'black',
            fontSize: [0, 0, 0, 1],
            fontFamily: 'heading',
            textTransform: 'uppercase',
            letterSpacing: 'mono',
            display: ['none', 'block', 'block', 'block'],
          }}
        >
          <Colorbar
            horizontal={true}
            bottom={true}
            colormap={range(0, 1, 0.1).map(
              candidateColorScales[selectedCandidate],
            )}
            clim={[0, 1]}
            format={(d) => `${d * 100}%`}
          />
        </Box>
      )}
      <Box
        sx={{
          position: 'absolute',
          bottom: [2, 5, 5, 5],
          left: 5,
          zIndex: 1000,
          borderRadius: 2,
          display: ['none', 'none', 'initial', 'initial'],
        }}
      >
        <Link href='https://socialists.nyc/'>
          <Image
            sx={{ mb: ['-12px'], mr: [5], width: 200 }}
            src='/logos/nycdsa-square-transparent.png'
          />
        </Link>
      </Box>
      <Box
        sx={{
          position: 'absolute',
          bottom: ['28px', 6, 6, 6],
          left: ['initial', 3, 'initial', 'initial'],
          right: [4, 'initial', 3, 3],
          zIndex: 1000,
          borderRadius: 2,
        }}
      >
        <Image
            src='/logos/qrcode_socialists.nyc.png'
            sx={{
              display: 'block',
              mx: 'auto',
              mt: 2,
              mb: 2,
              ml: [2, 5, 5, 5],
              width: ['155px', '155px'],
              display: ['none', 'none', 'block', 'block'],
            }}
          />
        <Link
          href='https://socialists.nyc/join?source=results'
          sx={{
            textAlign: 'center',
            ml: [2, 5, 5, 5],
            fontSize: [20, 32, 32, 32],
            fontFamily: 'heading',
            bg: 'black',
            borderRadius: '2px',
            color: 'white',
            px: ['10px', '10px', '10px', '10px'],
            pt: [1],
            pb: ['6px'],
            '&:hover': {
              color: 'rgb(200,200,200)',
              textDecoration: 'none',
            },
          }}
        >
          JOIN DSA
        </Link>
      </Box>
      <Box
        id='map'
        sx={{
          touchAction: 'pan-x pan-y',
          userSelect: 'none',
          zIndex: -1,
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100vh',
        }}
      ></Box>
    </>
  )
}

export default Index
